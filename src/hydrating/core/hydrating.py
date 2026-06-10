# -*- coding: utf-8 -*-
"""
Core rating-curve objects.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, TypeAlias, cast

import numpy as np
import pandas as pd

BooleanMask: TypeAlias = np.ndarray | pd.Series | list[bool]
WherePredicate: TypeAlias = Callable[[pd.Series], BooleanMask]
WhereCondition: TypeAlias = Hashable | WherePredicate


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
        If not provided, a column named "enabled" is added to the data with a boolean mask indicating which rows are active.
    metadata : dataframe or series-like, optional
        Additional metadata retained with the stage-discharge pairs, for example, method, date and time, and so on.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        h: str,
        q: str,
        enabled: str = "enabled",
        metadata: pd.DataFrame | pd.Series | list[Any] | np.ndarray | None = None,
    ):
        self.data = self._copy_data(data)
        self.h_col = h
        self.q_col = q
        self.enabled_col = enabled
        self.metadata = metadata
        self.fits = OrderedDict()

        self._validate_required_columns()
        if self.enabled_col not in self.data.columns:
            self.data[self.enabled_col] = True
        initial_enabled_mask = self._coerce_boolean_mask(
            self.data[self.enabled_col], name=self.enabled_col
        )
        self.data[self.enabled_col] = initial_enabled_mask.to_numpy()
        self._initial_enabled_mask = initial_enabled_mask.copy()

    @property
    def active_data(self) -> pd.DataFrame:
        """Rows currently enabled for fitting."""
        return self.data.loc[self.data[self.enabled_col]].copy()

    def enable(self, mask: BooleanMask) -> RatingCurve:
        """
        Replace the enabled row mask.

        Parameters
        ----------
        mask : array-like of bool
            Boolean mask with one value per row in `data`. `True` values
            mark rows as active for future fits.

        Returns
        -------
        RatingCurve
            This rating curve.

        Raises
        ------
        ValueError
            If the mask length does not match the data length, contains
            missing values, or is not boolean.
        """
        self.data[self.enabled_col] = self._coerce_boolean_mask(mask).to_numpy()
        return self

    def enable_where(self, **conditions: WhereCondition) -> RatingCurve:
        """
        Enable rows that match all provided column conditions.

        Parameters
        ----------
        **conditions
            Column filters. Non-callable values are matched exactly with
            ``data[column] == value``. Callable values are called with the full
            column and must return a boolean mask with one value per row.

        Returns
        -------
        RatingCurve
            This rating curve.

        Raises
        ------
        KeyError
            If a referenced column does not exist.
        ValueError
            If a callable condition returns an invalid boolean mask.
        """
        mask = pd.Series(True, index=self.data.index)

        for column, condition in conditions.items():
            values = self.data[column]
            if callable(condition):
                predicate = cast(WherePredicate, condition)  # for type checking
                condition_mask = self._coerce_boolean_mask(
                    predicate(values), name=f"condition for {column!r}"
                )
            else:
                condition_mask = self._coerce_boolean_mask(
                    values == condition, name=f"condition for {column!r}"
                )
            mask &= condition_mask

        self.data[self.enabled_col] = mask.to_numpy()

        return self

    def enable_all(self) -> RatingCurve:
        """
        Enable every row in the rating-curve data.

        Returns
        -------
        RatingCurve
            This rating curve.
        """
        self.data[self.enabled_col] = True

        return self

    def enable_none(self) -> RatingCurve:
        """
        Disable every row in the rating-curve data.

        Returns
        -------
        RatingCurve
            This rating curve.
        """
        self.data[self.enabled_col] = False

        return self

    def enable_reset(self) -> RatingCurve:
        """
        Restore the enabled mask that was passed when the rating curve was created.

        Returns
        -------
        RatingCurve
            This rating curve.
        """
        self.data[self.enabled_col] = self._initial_enabled_mask.to_numpy()

        return self

    @staticmethod
    def _copy_data(data) -> pd.DataFrame:
        if hasattr(data, "copy"):
            return data.copy()
        return pd.DataFrame(data)

    def _validate_required_columns(self):
        missing = [
            column for column in (self.h_col, self.q_col) if column not in self.data
        ]
        if missing:
            raise KeyError(f"Missing required data column(s): {missing}")

    def _coerce_boolean_mask(
        self, mask: BooleanMask, *, name: str = "mask"
    ) -> pd.Series:
        """
        Coerce and validate a row mask against the rating-curve data length.

        Parameters
        ----------
        mask : array-like of bool
            Boolean mask.
        name : str, default "mask"
            Name used in validation error messages only.

        Returns
        -------
        pandas.Series
            Boolean mask indexed like `data`.

        Raises
        ------
        ValueError
            If the mask length does not match the data length, contains
            missing values, or is not boolean.
        """
        mask_series = pd.Series(mask)
        if len(mask_series) != len(self.data):
            raise ValueError(
                f"{name} length must match data length "
                f"({len(self.data)}), got {len(mask_series)}."
            )
        if mask_series.isna().any():
            raise ValueError(f"{name} must not contain missing values.")
        if not pd.api.types.is_bool_dtype(mask_series):
            raise ValueError(f"{name} must contain boolean values.")
        return pd.Series(mask_series.to_numpy(dtype=bool), index=self.data.index)
