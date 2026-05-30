"""Categorical chunk sampler.

:class:`CategoricalSampler` draws fixed-size chunks from a dataset described by a
categorical column (one integer code per observation). Every chunk it yields lies
entirely within a single category, so each on-disk read stays contiguous and the
category label of a chunk is known without an extra lookup.

The category layout is stored as a run-length encoding (the maximal contiguous
runs of equal code) in a few numpy integer arrays, and a whole epoch's chunks are
drawn in one vectorized pass. Memory therefore scales with the number of runs
(at most ``n_obs / chunk_size``) and is independent of the number of categories.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.types import LoadRequest


class CategoricalSampler(Sampler):
    """Sampler that draws category-coherent chunks from a categorical column.

    The dataset is described by ``codes``, an integer category code per
    observation. A category may occupy any number of contiguous runs along the
    observation axis (it need not be a single block). On every draw the sampler
    picks a category, then a uniformly random ``chunk_size``-wide window that sits
    entirely inside one of that category's runs. As a result every yielded chunk
    contains observations of exactly one category.

    The distribution *within* a category is uniform over its valid chunk-start
    positions. The distribution *over* categories is uniform by default and can be
    reshaped with ``category_weights`` -- for example, pass each category's
    observation count to sample categories in proportion to their size.

    Chunks are drawn independently (i.e. with replacement); ``num_samples`` sets
    how many observations make up one epoch.

    Run-length rule
    ---------------
    Every contiguous run of a sampled category must be at least ``chunk_size``
    observations long. A run shorter than ``chunk_size`` can never contain a full
    chunk, so rather than silently ignore it the sampler raises at construction
    and names the offending categories. Re-chunk the data so each category's runs
    are large enough, lower ``chunk_size``, or exclude those categories via
    ``categories``.

    Multiple data-loading workers are not supported.

    Parameters
    ----------
    chunk_size
        Number of observations in each chunk (the width of a single read).
    preload_nchunks
        Number of chunks loaded into memory per iteration.
    batch_size
        Number of observations per batch.
    codes
        1D integer array with one category code per observation, e.g.
        ``df["cell_type"].cat.codes``. Its length must equal the loader's ``n_obs``.
    num_samples
        Total number of observations to draw per epoch.
    categories
        Optional subset of category codes to sample from. When ``None`` (the
        default) every category present in ``codes`` is sampled. Requested codes
        must exist in ``codes``, and the run-length rule is only enforced for the
        selected categories.
    category_weights
        Optional non-negative weights, aligned with :attr:`categories`, setting
        how often each category is drawn. When ``None`` (the default) categories
        are drawn uniformly. For size-proportional sampling pass the per-category
        observation counts, e.g. ``np.bincount(codes)[sampler.categories]``.
    drop_last
        Whether to drop the final incomplete batch.
    rng
        Random number generator. :func:`torch.manual_seed` has no effect here;
        pass a seeded :class:`numpy.random.Generator` for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> from annbatch.samplers import CategoricalSampler
    >>> codes = np.array([0] * 1000 + [1] * 1000 + [0] * 1000)  # category 0 is non-contiguous
    >>> sampler = CategoricalSampler(
    ...     chunk_size=256,
    ...     preload_nchunks=4,
    ...     batch_size=64,
    ...     codes=codes,
    ...     num_samples=3000,
    ... )
    """

    _batch_size: int
    _chunk_size: int
    _preload_nchunks: int
    _in_memory_size: int
    _num_samples: int

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        codes: np.ndarray,
        num_samples: int,
        categories: np.ndarray | None = None,
        category_weights: np.ndarray | None = None,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        self._validate_sizes(chunk_size, preload_nchunks, batch_size)
        check_lt_1([num_samples], ["num_samples"])

        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError("codes must be a 1D array of category codes (one per observation).")

        self._n_obs = int(codes.shape[0])
        self._rng = rng or np.random.default_rng()
        self._num_samples = num_samples
        self._drop_last = drop_last
        self._batch_size, self._chunk_size, self._preload_nchunks = batch_size, chunk_size, preload_nchunks
        self._in_memory_size = chunk_size * preload_nchunks

        self._build_runs(codes, chunk_size, categories)
        self._build_category_probs(category_weights)

    @staticmethod
    def _validate_sizes(chunk_size: int, preload_nchunks: int, batch_size: int) -> None:
        check_lt_1([chunk_size, preload_nchunks, batch_size], ["chunk_size", "preload_nchunks", "batch_size"])
        preload_size = chunk_size * preload_nchunks
        if batch_size > preload_size:
            raise ValueError(
                "batch_size cannot exceed chunk_size * preload_nchunks. "
                f"Got batch_size={batch_size}, but max is {preload_size}."
            )
        if preload_size % batch_size != 0:
            raise ValueError(
                "chunk_size * preload_nchunks must be divisible by batch_size. "
                f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
            )

    def _build_runs(self, codes: np.ndarray, chunk_size: int, categories: np.ndarray | None) -> None:
        """Run-length encode the codes and precompute per-category chunk-start offsets.

        ``_cum`` is the prefix-sum, across all (selected) runs sorted by category, of the
        number of valid chunk-start positions each run offers (``run_len - chunk_size + 1``).
        ``_cat_base`` / ``_cat_total`` carve out, for each category, the contiguous span it
        owns inside ``_cum`` -- which is what makes a single vectorized draw across all
        categories possible.
        """
        # maximal contiguous runs of identical category code
        boundaries = np.flatnonzero(np.diff(codes)) + 1
        run_start = np.concatenate([np.array([0], dtype=np.int64), boundaries.astype(np.int64)])
        run_len = np.concatenate([boundaries, np.array([codes.shape[0]])]).astype(np.int64) - run_start
        run_cat = codes[run_start]

        # restrict to the requested subset of categories, if any
        if categories is not None:
            categories = np.asarray(categories)
            missing = np.setdiff1d(categories, np.unique(codes))
            if missing.size:
                raise ValueError(f"Requested categories {missing.tolist()} are not present in codes.")
            keep = np.isin(run_cat, categories)
            run_start, run_len, run_cat = run_start[keep], run_len[keep], run_cat[keep]
        if run_cat.size == 0:
            raise ValueError("No categories to sample from.")

        # run-length rule: every (selected) run must be able to host a full chunk
        too_short = run_len < chunk_size
        if np.any(too_short):
            bad = np.unique(run_cat[too_short])
            raise ValueError(
                f"Every contiguous run must be at least chunk_size ({chunk_size}) observations long, "
                f"but {int(too_short.sum())} run(s) are shorter (categories {bad.tolist()}). "
                "Re-chunk the data so each category's runs are large enough, lower chunk_size, "
                "or exclude these categories."
            )
        n_pos = run_len - chunk_size + 1  # valid chunk-start positions per run

        # group runs by category so each category owns a contiguous span of `_cum`
        order = np.argsort(run_cat, kind="stable")
        run_start, run_cat, n_pos = run_start[order], run_cat[order], n_pos[order]

        cum = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(n_pos)])
        cat_ids, first = np.unique(run_cat, return_index=True)
        last = np.append(first[1:], len(run_cat))

        self._run_start = run_start
        self._cum = cum
        self._cat_ids = cat_ids
        self._cat_base = cum[first]  # offset into `_cum` where each category begins
        self._cat_total = cum[last] - cum[first]  # number of valid chunk positions per category

    def _build_category_probs(self, category_weights: np.ndarray | None) -> None:
        if category_weights is None:
            weights = np.ones(self._cat_ids.shape, dtype=float)
        else:
            weights = np.asarray(category_weights, dtype=float)
            if weights.shape != self._cat_ids.shape:
                raise ValueError(
                    f"category_weights must align with categories (expected shape {self._cat_ids.shape}, "
                    f"got {weights.shape}). See the `categories` property for the expected order."
                )
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("category_weights must be non-negative and not all zero.")
        self._probs = weights / weights.sum()

    @property
    def categories(self) -> np.ndarray:
        """Category codes this sampler draws from, in the order ``category_weights`` expects."""
        return self._cat_ids

    @property
    def mask(self) -> slice:
        raise NotImplementedError(
            "mask is not implemented for CategoricalSampler since it operates on a categorical column."
        )

    @mask.setter
    def mask(self, value: slice) -> None:
        raise NotImplementedError(
            "mask is not implemented for CategoricalSampler since it operates on a categorical column."
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    def n_iters(self, n_obs: int) -> int:
        del n_obs  # the epoch length is set by num_samples, not the loader size
        return (
            self._num_samples // self.batch_size if self._drop_last else math.ceil(self._num_samples / self.batch_size)
        )

    def validate(self, n_obs: int) -> None:
        """Check that the codes describe exactly the loader's observations."""
        if n_obs != self._n_obs:
            raise ValueError(
                f"codes length ({self._n_obs}) does not match loader n_obs ({n_obs}). "
                "The categorical column must describe exactly the loader's observations."
            )

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        del n_obs  # nothing is inferred from n_obs
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Multiple workers are not supported with CategoricalSampler.")
        yield from self._iter_from_chunks(self._compute_chunks())

    def _compute_chunks(self) -> list[slice]:
        """Draw the chunks for one epoch in a single vectorized pass."""
        n_chunks, remainder = divmod(self._num_samples, self._chunk_size)
        if remainder > 0 and not self._drop_last:
            n_chunks += 1

        # 1) choose a category for each chunk (uniform by default, weighted otherwise)
        cat_of_draw = self._rng.choice(len(self._cat_ids), size=n_chunks, p=self._probs)

        # 2) pick a uniform position within the chosen category's valid chunk starts
        local_off = (self._rng.random(n_chunks) * self._cat_total[cat_of_draw]).astype(np.int64)
        global_off = self._cat_base[cat_of_draw] + local_off

        # 3) translate the flat position into an absolute chunk-start row via the run it lands in
        run_idx = np.searchsorted(self._cum, global_off, side="right") - 1
        within = global_off - self._cum[run_idx]
        chunk_starts = self._run_start[run_idx] + within
        # the category label of each chunk is self._cat_ids[cat_of_draw], available if needed

        chunks = [slice(int(s), int(s + self._chunk_size)) for s in chunk_starts]
        if remainder > 0 and not self._drop_last:
            last = int(chunk_starts[-1])
            chunks[-1] = slice(last, last + remainder)
        return chunks

    def _iter_from_chunks(self, chunks: list[slice]) -> Iterator[LoadRequest]:
        """Group chunks into preload-sized requests and split each into shuffled batches."""
        chunks_per_request = split_given_size(chunks, self._preload_nchunks)
        batch_indices = np.arange(self._in_memory_size)
        for request_chunks in chunks_per_request[:-1]:
            # in-place shuffle so consecutive requests do not reuse the same batch order
            self._rng.shuffle(batch_indices)
            yield {"chunks": request_chunks, "splits": split_given_size(batch_indices, self._batch_size)}

        # the final request may hold fewer observations than a full in-memory window
        final_chunks = chunks_per_request[-1]
        total = int(sum(s.stop - s.start for s in final_chunks))
        if total == 0:  # pragma: no cover
            raise RuntimeError("Last request was found to have no observations. Please open an issue.")
        if self._drop_last:
            if total < self._batch_size:
                return
            total -= total % self._batch_size
        indices = self._rng.permutation(total)
        yield {"chunks": final_chunks, "splits": split_given_size(indices, self._batch_size)}
