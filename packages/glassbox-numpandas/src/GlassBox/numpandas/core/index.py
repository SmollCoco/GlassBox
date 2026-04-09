"""Index implementation for numpandas."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np


class Index:
    """Lightweight index for label-based selection.

    Parameters
    ----------
    labels : Iterable[Any]
        Index labels.
    """

    def __init__(self, labels: Iterable[Any]):
        if labels is None:
            raise ValueError("Index labels must not be None.")
        self._labels = np.asarray(list(labels), dtype=object)
        self._pos_cache: dict[Any, int] | None = None

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __iter__(self):
        return iter(self._labels)

    def __getitem__(self, key):
        return self._labels[key]

    def to_list(self) -> list[Any]:
        """Return labels as a list.

        Returns
        -------
        list
            Labels as Python list.
        """
        return self._labels.tolist()

    def get_loc(self, label: Any) -> int:
        """Return integer position for a label.

        Parameters
        ----------
        label : Any
            Label to locate.

        Returns
        -------
        int
            Integer location.

        Raises
        ------
        KeyError
            If label is not present.
        ValueError
            If duplicate labels exist.
        """
        if self._pos_cache is None:
            positions: dict[Any, int] = {}
            for idx, value in enumerate(self._labels):
                if value in positions:
                    raise ValueError("Index contains duplicate labels.")
                positions[value] = idx
            self._pos_cache = positions
        if label not in self._pos_cache:
            raise KeyError(f"Label not in index: {label}")
        return self._pos_cache[label]
