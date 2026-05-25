# -*- coding: utf-8 -*-
"""
Core rating-curve objects.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Fit:
    """Named fit result."""

    name: str


class LmfitBackend:
    """Internal lmfit backend."""

    name = "lmfit"

    def fit(self, *, model, h, q, uncertainty=None, **options):
        """Fit a model using lmfit."""
        raise NotImplementedError("LmfitBackend.fit() is not implemented yet.")

# TODO should the uncertainty column be set here? (can be None)
class RatingCurve:
    """
    Store rating-curve measurement data and named fits.

    Parameters
    ----------
    data : dataframe-like
        Measurement dataframe containing at least the stage and discharge columns.
    h : str
        Name of the stage column.
    q : str
        Name of the discharge column.
    enabled : str, default "enabled"
        Name of the column controlling whether rows are active, meaning if a measurement should be used to fit the model.
    metadata : Any, optional
        Additional metadata retained with the stage-discharge pairs, for example, method, date and time, and so on.
    """

    def __init__(
        self,
        data,
        h: str,
        q: str,
        enabled: str = "enabled",
        metadata: Any | None = None,
    ):
        self.data = self._copy_data(data)
        self.h = h
        self.q = q
        self.enabled = enabled
        self.metadata = metadata
        self.fits = OrderedDict()

        self._validate_required_columns()
        if self.enabled not in self.data.columns:
            self.data[self.enabled] = True

    @property
    def active_data(self):
        """Rows currently enabled for fitting."""
        return self.data.loc[self.data[self.enabled]].copy()

    @staticmethod
    def _copy_data(data):
        if hasattr(data, "copy"):
            return data.copy()
        return pd.DataFrame(data)

    def _validate_required_columns(self):
        missing = [column for column in (self.h, self.q) if column not in self.data]
        if missing:
            raise KeyError(f"Missing required data column(s): {missing}")

