"""Categorical sampler for category-stratified data access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from annbatch.sampler._sampler import Sampler, SliceSampler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pandas as pd


class CategoricalSampler(Sampler[list[slice]]):
    """Category-stratified sampler using composition with SliceSampler.

    This sampler maintains one SliceSampler per category and yields from each
    in turn. Each batch contains observations from only one category.

    Parameters
    ----------
    obs
        DataFrame containing categorical columns to stratify by.
    keys
        Column name(s) in `obs` to use for categorical stratification.
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk in the backing store.
    preload_nchunks
        Number of chunks to load per iteration.
    shuffle
        Whether to shuffle category order and within-category data.
    drop_last
        Whether to drop the last incomplete batch per category.
    category_weights
        Optional weights for sampling category order when shuffle=True.
        If None, categories are sampled uniformly. Must sum to 1.0.
    rng
        Random number generator for shuffling and category sampling.

    Examples
    --------
    >>> import pandas as pd
    >>> obs = pd.DataFrame(
    ...     {
    ...         "cell_type": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
    ...     }
    ... )
    >>> sampler = CategoricalSampler(
    ...     obs=obs,
    ...     keys="cell_type",
    ...     batch_size=2,
    ...     chunk_size=10,
    ...     preload_nchunks=1,
    ... )
    >>> for slices, splits, leftover in sampler:
    ...     # Each batch contains only one category
    ...     print(slices)

    Notes
    -----
    **Data requirements**: The data must be sorted/grouped such that each
    unique combination of the specified keys forms a contiguous block.
    This is validated on initialization.

    **Potential issues with this design**:

    1. **Data loss with leftovers**: Each category's SliceSampler manages its
       own leftovers independently. When `drop_last=True`, partial batches
       at category boundaries are dropped. Even with `drop_last=False`,
       leftovers are yielded at the end of each category, not carried across
       categories. Consider: Is there a row that won't be seen by the model?

    2. **Chunk loading inefficiency**: If category boundaries don't align with
       chunk boundaries. Chunks are loaded multiple times. But it's
       not a problem in certain cases.

    3. **Randomness considerations**: With `shuffle=True`, the loop
    structure might have some pattern that might affect training.

    To fix this leftover issue, samplers will need to access the
    loaded data. Which might not be desirable in some cases.
    So think about this.
    """

    __slots__ = (
        "_obs",
        "_keys",
        "_batch_size",
        "_chunk_size",
        "_preload_nchunks",
        "_shuffle",
        "_drop_last",
        "_category_weights",
        "_rng",
        "_category_bounds",
        "_category_labels",
        "_category_samplers",
        "_total_batches",
    )

    def __init__(
        self,
        *,
        obs: pd.DataFrame,
        keys: str | Sequence[str],
        batch_size: int,
        chunk_size: int,
        preload_nchunks: int,
        shuffle: bool = False,
        drop_last: bool = False,
        category_weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self._obs = obs
        self._keys = [keys] if isinstance(keys, str) else list(keys)
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = np.random.default_rng() if rng is None else rng

        # Validate keys exist in obs
        for key in self._keys:
            if key not in obs.columns:
                msg = f"Key '{key}' not found in obs columns"
                raise ValueError(msg)

        # Compute category boundaries and validate contiguity
        bounds, labels = self._compute_category_bounds()
        self._category_bounds = bounds
        self._category_labels = labels

        # Create one SliceSampler per category
        self._category_samplers = self._create_category_samplers()

        # Compute total batches
        self._total_batches = sum(len(s) for s in self._category_samplers)

        # Validate and set category weights
        n_categories = len(self._category_bounds)
        if category_weights is not None:
            if len(category_weights) != n_categories:
                msg = (
                    f"category_weights length ({len(category_weights)}) "
                    f"must match number of categories ({n_categories})"
                )
                raise ValueError(msg)
            if not np.isclose(np.sum(category_weights), 1.0):
                weights_sum = np.sum(category_weights)
                msg = f"category_weights must sum to 1.0, got {weights_sum}"
                raise ValueError(msg)
            self._category_weights = np.asarray(category_weights)
        else:
            self._category_weights = None

    def _compute_category_bounds(
        self,
    ) -> tuple[list[tuple[int, int]], list[tuple]]:
        """Compute start/end indices for each category and validate contiguity.

        Returns
        -------
        bounds
            List of (start_index, end_index) tuples for each category.
        labels
            List of category label tuples corresponding to each bound.

        Raises
        ------
        ValueError
            If observations are not contiguously grouped by category.
        """
        if len(self._keys) == 1:
            categories = self._obs[self._keys[0]].values
        else:
            # Create tuple of values for multi-key grouping
            categories = list(zip(*[self._obs[k].values for k in self._keys], strict=True))

        bounds: list[tuple[int, int]] = []
        labels: list[tuple] = []
        seen_categories: set = set()

        if len(categories) == 0:
            return bounds, labels

        current_cat = categories[0]
        current_start = 0

        for i, cat in enumerate(categories[1:], start=1):
            if cat != current_cat:
                # End of current category block
                if isinstance(current_cat, tuple):
                    cat_key = current_cat
                else:
                    cat_key = (current_cat,)
                if cat_key in seen_categories:
                    msg = (
                        f"Category {current_cat} appears non-contiguously at "
                        f"index {i}. Data must be grouped by category. "
                        f"Sort your data by {self._keys} before creating the "
                        "sampler."
                    )
                    raise ValueError(msg)

                bounds.append((current_start, i))
                labels.append(cat_key)
                seen_categories.add(cat_key)

                current_cat = cat
                current_start = i

        # Handle last category
        if isinstance(current_cat, tuple):
            cat_key = current_cat
        else:
            cat_key = (current_cat,)
        if cat_key in seen_categories:
            msg = (
                f"Category {current_cat} appears non-contiguously. "
                f"Data must be grouped by category. "
                f"Sort your data by {self._keys} before creating the sampler."
            )
            raise ValueError(msg)
        bounds.append((current_start, len(categories)))
        labels.append(cat_key)

        return bounds, labels

    def _create_category_samplers(self) -> list[SliceSampler]:
        """Create one SliceSampler per category."""
        samplers = []
        for start, end in self._category_bounds:
            sampler = SliceSampler(
                n_obs=self._obs.shape[0],
                batch_size=self._batch_size,
                chunk_size=self._chunk_size,
                preload_nchunks=self._preload_nchunks,
                shuffle=self._shuffle,
                drop_last=self._drop_last,
                start_index=start,
                end_index=end,
                rng=self._rng,
            )
            samplers.append(sampler)
        return samplers

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(
        self,
    ) -> Iterator[tuple[list[slice], list[np.ndarray], None]]:
        """Iterate by yielding from each category's sampler in turn.

        The iteration interleaves batches from different categories.
        Each batch contains observations from only one category.

        Yields
        ------
        slices
            List of slices to load from the backing store.
        splits
            List of index arrays for batching the loaded data.
        leftover
            Always None (leftovers are not carried across categories).
        """
        n_categories = len(self._category_samplers)
        if n_categories == 0:
            return

        # Create iterators for each category
        category_iters = [iter(s) for s in self._category_samplers]
        active = list(range(n_categories))

        # Shuffle or use weights for category order if requested
        if self._shuffle:
            if self._category_weights is not None:
                # Reorder based on weights (higher weight = earlier in shuffle)
                active = list(
                    self._rng.choice(
                        active,
                        size=len(active),
                        replace=False,
                        p=self._category_weights,
                    )
                )
            else:
                self._rng.shuffle(active)

        # Round-robin through categories, yielding one batch at a time
        while active:
            next_active = []
            for cat_idx in active:
                try:
                    slices, splits, _ = next(category_iters[cat_idx])
                    yield slices, splits, None
                    next_active.append(cat_idx)
                except StopIteration:
                    # This category is exhausted
                    pass
            active = next_active

            # Re-shuffle remaining categories if shuffle is enabled
            if self._shuffle and active:
                if self._category_weights is not None:
                    weights = self._category_weights[active]
                    weights = weights / weights.sum()
                    active = list(self._rng.choice(active, size=len(active), replace=False, p=weights))
                else:
                    self._rng.shuffle(active)

    @property
    def category_bounds(self) -> list[tuple[int, int]]:
        """Return the (start, end) bounds for each category."""
        return self._category_bounds

    @property
    def category_labels(self) -> list[tuple]:
        """Return the category labels corresponding to each bound."""
        return self._category_labels

    @property
    def n_categories(self) -> int:
        """Return the number of unique categories."""
        return len(self._category_bounds)

    @property
    def category_samplers(self) -> list[SliceSampler]:
        """Return the list of SliceSamplers, one per category."""
        return self._category_samplers
