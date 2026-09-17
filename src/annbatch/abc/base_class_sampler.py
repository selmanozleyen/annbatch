from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from annbatch.abc.sampler import Sampler

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class BaseClassSampler(Sampler):
    """Abstract interface for class-coherent samplers."""

    @property
    @abstractmethod
    def vocab(self) -> pd.Index:
        """Label vocabulary the emitted codes index into."""

    @abstractmethod
    def emittable_codes(self) -> np.ndarray:
        """Codes into :attr:`vocab` this sampler can draw."""

    @abstractmethod
    def batch_codes(self) -> np.ndarray:
        """Class code of each batch a full pass yields. Advances the related rng."""
