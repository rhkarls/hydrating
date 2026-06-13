# -*- coding: utf-8 -*-
"""
Core rating-curve objects.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
    model: Any
    backend: str
    result_: Any
    active_data_: pd.DataFrame
    h_col: str
    q_col: str
    params_: dict[str, float] = field(init=False)
    derived_params_: dict[str, float] = field(init=False)
    aic: float = field(init=False)
    bic: float = field(init=False)
    redchi: float = field(init=False)
    reduced_chi: float = field(init=False)
    r2: float = field(init=False)
    mean_absolute_error: float = field(init=False)
    mean_percentage_error: float = field(init=False)
    mean_absolute_percentage_error: float = field(init=False)

    def __post_init__(self):
        self.active_data_ = self.active_data_.copy(deep=True)
        self.params_ = {
            name: float(parameter.value)
            for name, parameter in self.result_.params.items()
        }
        self.derived_params_ = self._derived_parameters()
        self.aic = float(self.result_.aic)
        self.bic = float(self.result_.bic)
        self.redchi = float(self.result_.redchi)
        self.reduced_chi = self.redchi
        self.r2 = float(self.result_.rsquared)
        self._calculate_metrics()

    def predict(self, h):
        """
        Predict discharge for stage values using fitted parameters.

        Parameters
        ----------
        h : array-like
            Stage values.

        Returns
        -------
        np.ndarray
            Predicted discharge values.
        """
        h_values = np.asarray(h, dtype=float)
        return np.asarray(self.model.func(h_values, **self.params_), dtype=float)

    def _calculate_metrics(self):
        observed = self.active_data_[self.q_col].to_numpy(dtype=float)
        predicted = self.predict(self.active_data_[self.h_col].to_numpy(dtype=float))
        residuals = predicted - observed

        self.mean_absolute_error = float(np.mean(np.abs(residuals)))

        nonzero = observed != 0
        if np.any(nonzero):
            percentage_errors = residuals[nonzero] / observed[nonzero] * 100.0
            self.mean_percentage_error = float(np.mean(percentage_errors))
            self.mean_absolute_percentage_error = float(
                np.mean(np.abs(percentage_errors))
            )
        else:
            self.mean_percentage_error = np.nan
            self.mean_absolute_percentage_error = np.nan

    def _derived_parameters(self) -> dict[str, float]:
        if not hasattr(self.model, "derived_parameters"):
            return {}
        derived = self.model.derived_parameters(self.params_)
        return {name: float(value) for name, value in derived.items()}


# TODO move out of this file
class LmfitBackend:
    """Internal lmfit backend."""

    name = "lmfit"

    def fit(self, *, model, h, q, uncertainty=None, **options):
        """Fit a model using lmfit."""
        if uncertainty is not None:
            raise NotImplementedError(
                "uncertainty not yet implementation, pass as None."
            )

        h_values = np.asarray(h, dtype=float)
        q_values = np.asarray(q, dtype=float)
        parameters = model.constrain_parameters(h_values, q_values).copy()
        lmfit_model = model.create_lmfit_model()

        return lmfit_model.fit(q_values, params=parameters, h=h_values, **options)


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
        self.fits = OrderedDict()  # TODO maybe we can just use regular

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

    def fit(
        self,
        name: str,
        *,
        model,
        uncertainty=None,
        backend: str = "lmfit",
        overwrite: bool = False,
        **lmfit_options,
    ) -> Fit:
        """
        Fit a model to currently enabled stage-discharge pairs.

        Parameters
        ----------
        name : str
            Name used to store the fit.
        model
            Rating model with the hydrating model interface.
        uncertainty : optional
            Discharge gauging uncertainty. Currently not implemented.
        backend : {"lmfit"}, default "lmfit"
            Fitting backend.
        overwrite : bool, default False
            Whether an existing fit with the same name may be replaced.
        **lmfit_options
            Additional keyword arguments passed to ``lmfit.Model.fit``.

        Returns
        -------
        Fit
            A Fit object.
        """
        if backend != LmfitBackend.name:
            raise NotImplementedError(f"Unsupported backend: {backend!r}.")
        if name in self.fits and not overwrite:
            raise ValueError(f"Fit {name!r} already exists.")

        active_data = self.active_data
        backend_inst = LmfitBackend()
        result = backend_inst.fit(
            model=model,
            h=active_data[self.h_col].to_numpy(),
            q=active_data[self.q_col].to_numpy(),
            uncertainty=uncertainty,
            **lmfit_options,
        )
        fit = Fit(
            name=name,
            model=model,
            backend=backend,
            result_=result,
            active_data_=active_data,
            h_col=self.h_col,
            q_col=self.q_col,
        )
        self.fits[name] = fit

        return fit

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
