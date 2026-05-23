# -*- coding: utf-8 -*-
"""
RatingCurve class.
"""

from typing import Optional

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrating.models import RatingModel

# from .grades import Grades


class RatingCurve:
    """
    Represent and fit a hydrologic rating curve.

    A rating curve describes the relationship between stage (water level)
    and discharge (flow). This class wraps a rating-model implementation,
    manages initial parameter values, optionally stores observed data in a
    DataFrame, and fits the selected model using ``lmfit`` package.

    Parameters
    ----------
    model : RatingModel
        RatingModel class used to create the underlying model and
        parameters. The class is instantiated internally.
    initial_parameters : dict or lmfit.Parameters, optional
        Initial parameter values for the model. If a dictionary is
        provided, it is converted to ``lmfit.Parameters`` using the
        selected rating model. If omitted, default parameters are created.
        It is recommended to provide initial parameter values for the model,
        as they are often needed for successful fitting.
    rating_name : str, optional
        Optional name identifier for the rating curve.

    Attributes
    ----------
    rating_name : str
        User supplied name for the rating curve.
    rating_model : RatingModel
        Instantiated rating-model object used for fitting and prediction.
    initial_parameters : lmfit.Parameters
        Parameter set used as the starting point for fitting.
    fit_result : lmfit.model.ModelResult or None
        Result returned by the most recent fit.
    stage_series : pandas.Series or numpy.ndarray or None
        Optional stage time series used for prediction of discharge.
    """

    def __init__(
        self,
        model: RatingModel,
        initial_parameters: dict | lmfit.Parameters | None = None,
        rating_name: str = "",
    ):
        """
        Initialize a rating curve.

        Parameters
        ----------
        model : RatingModel
            RatingModel class used to create the underlying model instance,
            parameters, fitted values, and predictions. The class is
            instantiated internally.
        initial_parameters : dict or lmfit.Parameters, optional
            Initial parameter values for the rating model. Dictionaries are
            converted to ``lmfit.Parameters`` by the selected rating model. If
            omitted, the rating model's default parameters are used. It is recommended
            to provide initial parameter values for the rating model, as they are often
            needed for successful fitting.
        rating_name : str, optional
            Name identifier for the rating curve.

        Raises
        ------
        ValueError
            If ``initial_parameters`` is not a dictionary, ``lmfit.Parameters``,
            or ``None``.
        """

        self.rating_name = rating_name
        self.rating_model = model  # instance is created in the setter
        self.initial_parameters = initial_parameters  # done by model setter

        # self.rc_grade = None # grade stage intervals with Enum grades
        # def set_grades():
        #     ...# @setter/getter properties?

        # self.rc_limit = None # tuple, min, max stage its applicable
        # self.rc_period = None # tuple (datetime-like to from validity)

        self.fit_result = None

        self.stage_series = (
            None  # timeseries of stage, can be used to show extrapolation
        )
        # of the rating curve, when comparing rating curves

        self._dataf = None  # user can add a dataframe using add_data, but this is strictly not needed
        # can also just call .fit() with two numpy arrays
        self._user_df_added = False

        # def set_limits():
        #     ...  #@setter/getter properties?
        # def set_valid_periods():
        #     ...  #@setter/getter properties?

    # or use property??
    # rc.model = Callable
    # also for parameter
    # rc.parameter = dict # overwrites the default
    @property
    def rating_model(self):  # numpydoc ignore=GL08
        return self._rating_model

    @rating_model.setter
    def rating_model(self, model: RatingModel):  # numpydoc ignore=GL08
        self._rating_model = model()

    @property
    def initial_parameters(self):  # numpydoc ignore=GL08
        return self._initial_parameters

    @initial_parameters.setter
    def initial_parameters(self, params):
        """
        Set the initial parameters for the rating model.

        Parameters
        ----------
        params : dict or lmfit.Parameters or None
            If a dictionary is provided, it is converted to ``lmfit.Parameters``
            using the selected rating model. If omitted, default parameters are
            created. It is recommended to provide initial parameter values for the
            model, as they are often needed for successful fitting.

        Raises
        ------
        ValueError
            If ``params`` is not a dictionary, ``lmfit.Parameters``, or ``None``.
        """
        model = self.rating_model.create_model()
        if isinstance(params, dict):
            self._initial_parameters = self.rating_model.create_parameters(
                model, params
            )
        elif isinstance(params, lmfit.Parameters):
            self._initial_parameters = params
        elif params is None:
            self._initial_parameters = self.rating_model.create_parameters(model)
        else:
            raise ValueError()

        # Check the parameters

    # @property
    # def grade(self):
    #     return self._grade

    # @grade.setter
    # def grade(self, value):
    #     # [{'from': 1.5,
    #     #  'to': 2.5,
    #     #  'grade': Grades.Good},] ?
    #     self._grade = value

    def add_data(
        self,
        data: pd.DataFrame,
        stage: str,
        discharge: str,
        datetime_start: Optional[str] = None,  # optional, key in data df
        datetime_end: Optional[str] = None,  # optional, key in data df
        method: Optional[str] = None,  # optional, key in data df
        party: Optional[str] = None,  # optional, key in data df
        agency: Optional[str] = None,  # optional, key in data df
        uncertainty: Optional[str] = None,  # optional, key in data df
        grade: Optional[str] = None,  # optional, key in data df
        enabled: Optional[str] = None,  # optional, key in data df
        note: Optional[str] = None,  # optional, key in data df
        identifier: Optional[str] = None,
    ):  # optional, key in data df
        """
        Add stage-discharge measurements to the rating curve.

        Several keys for columns in the provided dataframe can be specified, but only
        the stage and discharge columns are required. The other columns are optional
        but can be used to store metadata. The ``enabled`` column is added automatically
        if not provided, and flags if a measurement is used or not in the fitting.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe containing observed stage, discharge, and optional
            metadata columns. A copy is stored internally.
        stage : str
            Column name in ``data`` containing stage values.
        discharge : str
            Column name in ``data`` containing discharge values.
        datetime_start : str, optional
            Column name in ``data`` containing observation start datetimes.
        datetime_end : str, optional
            Column name in ``data`` containing observation end datetimes.
        method : str, optional
            Column name in ``data`` containing measurement methods.
        party : str, optional
            Column name in ``data`` containing measurement parties.
        agency : str, optional
            Column name in ``data`` containing agencies responsible for the
            measurements.
        uncertainty : str, optional
            Column name in ``data`` containing measurement uncertainty values.
        grade : str, optional
            Column name in ``data`` containing observation grades.
        enabled : str, optional
            Column name in ``data`` containing flags for whether observations
            are enabled. If omitted, an ``enabled`` column with ``True`` values
            is added to the internally stored dataframe.
        note : str, optional
            Column name in ``data`` containing observation notes.
        identifier : str, optional
            Column name in ``data`` containing observation identifiers.
        """

        self._k_discharge = discharge
        self._k_stage = stage
        self._k_datetime_start = datetime_start
        self._k_datetime_end = datetime_end
        self._k_method = method
        self._k_party = party
        self._k_agency = agency
        self._k_uncertainty = uncertainty
        self._k_grade = grade
        self._k_enabled = enabled
        self._k_note = note
        self._k_identifer = identifier

        # take the data DataFrame and make a copy
        # keep all columns of DataFrame
        # create a new columns "enabled" for True/False flag to turn on off
        self._dataf = data.copy()
        if self._k_enabled is None:
            self._k_enabled = "enabled"
            self._dataf["enabled"] = True

        self._user_df_added = True

    # TODO use a setter/getter for dataf in addition to this function to add_data?

    def _user_data_from_fit(self, x, y):
        """
        If data is passed to fit() call then create _dataf here if the user
        has not done so themselves by calling add_data().

        FIXME: this needs improvement, will not work if user has added data with add_data()
        and then calls fit() with different data, it will overwrite the user added data.
        Maybe should check if _dataf is not None and raise an error if user is trying
        to add data through fit() when they have already added data with add_data()?

        Parameters
        ----------
        x : np.ndarray or pd.Series
            Stage data passed to fit() call.
        y : np.ndarray or pd.Series
            Discharge data passed to fit() call.
        """

        self._k_stage = "stage"
        self._k_discharge = "discharge"
        self._dataf = pd.DataFrame(data={self._k_stage: x, self._k_discharge: y})

    # fit uses the model and parameters
    # FIXME need to handle this better, if user has added data with add_data() and then calls fit() with different data, it will overwrite the user added data. Maybe should check if _dataf is not None and raise an error if user is trying to add data through fit() when they have already added data with add_data()?
    # data should be passed to fit() call, or when initialized maybe better
    # try to adopt a scikit-learn like API?
    # add_data should probably be removed, deprecate that
    def fit(
        self,
        x: pd.Series = None,  # FIXME np.ndarray, pd.Series or str for key in dataf
        y: pd.Series = None,
        engine: str = "lmfit",
        weights: pd.Series = None,
        **kwargs,
    ):
        """
        Fit the rating curve to observed stage-discharge data.

        Parameters
        ----------
        x : pandas.Series or numpy.ndarray or str, optional
            Stage values used for fitting. If a string is provided, it is used
            as a column name in the dataframe added with :meth:`add_data`. If
            omitted, the stage column configured by :meth:`add_data` is used.
        y : pandas.Series or numpy.ndarray or str, optional
            Discharge values used for fitting. If a string is provided, it is
            used as a column name in the dataframe added with :meth:`add_data`.
            If omitted, the discharge column configured by :meth:`add_data` is
            used.
        engine : {"lmfit"}, default: "lmfit"
            Fitting engine to use. Currently only ``"lmfit"`` is supported.
        weights : pandas.Series, optional
            Observation weights passed to ``lmfit.Model.fit``. If omitted, all
            observations are weighted equally.
        **kwargs
            Additional keyword arguments passed to ``lmfit.Model.fit``.

        Raises
        ------
        NotImplementedError
            If ``engine`` is not ``"lmfit"``.

        Notes
        -----
        The fitted result is stored in ``fit_result`` and the best-fit
        parameter values are stored in ``fit_best_parameters``. Fit residuals
        and percent errors are stored internally for plotting.
        """

        # Note: cannot use self. in default arguments as they are eval'ed at
        # creation time
        if x is None:
            # try:
            xf = self._dataf[self._k_stage].to_numpy()
            # except: # TODO what error is this if dataf is None?
            # ... # message to add data or pass data to fit
        elif isinstance(x, str):
            xf = self._dataf[x].to_numpy()
        else:  # is np.ndarray or Series, check for that?
            xf = x.copy()

        if y is None:
            yf = self._dataf[self._k_discharge].to_numpy()
        elif isinstance(y, str):
            yf = self._dataf[y].to_numpy()
        else:  # is np.ndarray or Series, check for that?
            yf = y.copy()

        if not self._user_df_added:
            self._user_data_from_fit(xf, yf)

        if engine != "lmfit":
            raise NotImplementedError(
                f"{engine} is not supported, only lmfit is currently supported"
            )

        # lmfit engine
        lmfit_model = self.rating_model.create_model()
        lmfit_init_pars = self.initial_parameters
        lmfit_weights = np.ones(len(yf)) if weights is None else weights.to_numpy()

        # constrain parameters as defined in RatingModel
        lmfit_init_pars = self.rating_model.constrain_pars_with_obs(
            xf, yf, lmfit_init_pars
        )

        result = lmfit_model.fit(
            data=yf, params=lmfit_init_pars, weights=lmfit_weights, h=xf, **kwargs
        )  # test with using kwargs = {'method':'differential_evolution'}

        self._fit_obs_data = pd.DataFrame(data=yf, index=xf, columns=["observed"])

        self.fit_result = result
        self.fit_best_parameters = result.best_values

        self._calc_fit_residuals()

    def _calc_fit_residuals(self):  # numpydoc ignore=GL08
        self._fit_obs_data["predicted"] = self.predict(stage=self._fit_obs_data.index)
        self._fit_obs_data["residual"] = (
            self._fit_obs_data["predicted"] - self._fit_obs_data["observed"]
        )
        self._fit_obs_data["percent_error"] = (
            self._fit_obs_data["residual"] / self._fit_obs_data["observed"] * 100
        )

    def add_stage_series(self, stage_series):  # numpydoc ignore=PR01
        """Set the stage series to be used for prediction and plotting."""
        self.stage_series = stage_series.copy()

    def predict(self, stage=None):
        """
        Predict discharge at a given stage or array-like of stages using the fitted model.

        Parameters
        ----------
        stage : float or array-like, optional
            Stage value(s) at which to predict discharge. If omitted, the stage series added
            with :meth:`add_stage_series` is used.
            If no stage series is available, ``None`` is returned.

        Returns
        -------
        float or np.ndarray
            Discharge values predicted at the given stage(s). If no stage series is available,
            ``None`` is returned.
        """
        if stage is None and self.stage_series is not None:
            return self.rating_model.func(self.stage_series, **self.fit_best_parameters)
        if stage is not None:
            return self.rating_model.func(stage, **self.fit_best_parameters)
        # FIXME raise
        return None

    def inverse(self, discharge, initial_guess=None):
        """
        Inverse the rating curve to find the stage at a given discharge.

        Parameters
        ----------
        discharge : float or array-like
            Discharge value(s) at which to predict stage.
        initial_guess : float or array-like, optional
            Initial guess for the stage value(s) corresponding to the given discharge value(s).
            This is required for models that do not have an inverse method.
        """
        raise NotImplementedError()
        # if model has inverse attribute, use that, else inverse it with minimizing
        # this requires an initial guess
        self.rating_model.inverse(
            discharge, initial_guess, **self.fit_result.best_values
        )

    def fit_summary(self):
        """Print a summary of the fitted model parameters and results."""
        print(self.fit_result.fit_report())

    def plot_residuals(
        self,
        scale="linear",
        label_discharge="Discharge",
        label_stage="Stage",
        stage_on_y=False,
        point_labels=None,
    ):
        """
        Plot residuals and model performance for a fitted rating curve.

        This function generates a plot to visualize the observed and predicted data, as well
        as the residuals and percentage errors for the rating curve model fit. It supports optional
        custom scaling for the axes and labeling.

        Parameters
        ----------
        scale : str, optional
            The scale type to be used for the y-axis of the observed/predicted plot and the x-axis
            of the residual and percent error plots. Default is "linear".
        label_discharge : str, optional
            Label for the discharge axis on the plots. Default is "Discharge".
        label_stage : str, optional
            Label for the stage axis on the plots. Default is "Stage".
        stage_on_y : bool, optional
            If True, plots stage on the y-axis and discharge on the x-axis. Default is False.
        point_labels : str, optional
            Column name in the fitted data DataFrame to use for labeling points in the plots. If
            provided, labels will be added to the observed data points in the first subplot.

        Returns
        -------
        tuple
            A tuple containing:
            - fig: matplotlib.figure.Figure
                The overall figure object for the plots.
            - axes: numpy.ndarray of matplotlib.axes.Axes
                An array of axes objects for the three subplots.

        Raises
        ------
        ValueError
            If the rating curve is not fitted before calling this method (i.e., `fit_result` is None).

        Notes
        -----
        - The first subplot displays the observed vs predicted data.
        - The second subplot shows the absolute residuals.
        - The third subplot illustrates the percent error.
        - If `stage_series` is provided, it defines the span of the stage axis; otherwise, the
          minimum and maximum observed stage values from the fitted data are used.
        - When `stage_on_y` is False (default), stage is the x-axis and discharge is the y-axis
          for the first subplot, while other subplots share the x-axis (stage).
        - Adjusts y-limits of the residual and percent error subplots for symmetric bounds.
        """
        if self.fit_result is None:
            ValueError(
                "Rating curve is not fitted, call .fit() first."
            )  # TODO custom exception

        if self.stage_series is not None:
            min_stage = self.stage_series.min()
            max_stage = self.stage_series.max()
        else:
            min_stage = self._fit_obs_data.index.min()
            max_stage = self._fit_obs_data.index.max()

        stage_plt = np.linspace(min_stage, max_stage, num=100)
        q_plt = self.predict(stage=stage_plt)

        fig, axes = plt.subplots(3, 1, sharex=True)

        if stage_on_y:
            y_plt = stage_plt
            x_plt = q_plt

            y_pts = self._fit_obs_data.index.to_numpy()
            x_pts = self._fit_obs_data["observed"]

            y_label = label_stage
            x_label = label_discharge
        else:
            y_plt = q_plt
            x_plt = stage_plt

            y_pts = self._fit_obs_data["observed"]
            x_pts = self._fit_obs_data.index.to_numpy()

            x_label = label_stage
            y_label = label_discharge

        axes[0].plot(x_pts, y_pts, "ko")
        axes[0].plot(x_plt, y_plt, "k-")

        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].plot(x_pts, self._fit_obs_data["residual"], "ko")

        axes[2].axhline(0, color="k", linewidth=0.5)
        axes[2].plot(x_pts, self._fit_obs_data["percent_error"], "ko")

        axes[0].set_yscale(scale)
        axes[2].set_xscale(scale)

        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel("Absolute error")
        axes[2].set_ylabel("Percent error")
        axes[2].set_xlabel(x_label)

        axes[1].set_ylim(
            (-max(np.abs(axes[1].get_ylim())), max(np.abs(axes[1].get_ylim())))
        )
        axes[2].set_ylim(
            (-max(np.abs(axes[2].get_ylim())), max(np.abs(axes[2].get_ylim())))
        )

        return fig, axes

    def plot(
        self,
        scale="linear",
        label_discharge="Discharge",
        label_stage="Stage",
        stage_on_y=False,
        point_labels=None,
        cross_section=False,
    ):
        """
        Plot the fitted rating curve and observational data.

        Parameters
        ----------
        scale : str, optional
            A string indicating the type of scaling for the axes. Default is "linear".
        label_discharge : str, optional
            Label for the discharge axis. Default is "Discharge".
        label_stage : str, optional
            Label for the stage axis. Default is "Stage".
        stage_on_y : bool, optional
            If True, stage is plotted along the y-axis; otherwise, discharge is plotted on
            the y-axis. Default is False.
        point_labels : str, optional
            Labels for the individual data points on the plot. Default is None.
        cross_section : bool, optional
            Plots the cross-section profile if available. If True, forces stage to be plotted on the y-axis, overriding the `stage_on_y`
            parameter. Default is False.

        Returns
        -------
        tuple
            Contains:
            - fig : matplotlib.figure.Figure
                The matplotlib figure object containing the plot.
            - ax : matplotlib.axes._axes.Axes
                The matplotlib axes object for the plot.

        Raises
        ------
        ValueError
            If the rating curve has not been fitted using the `.fit()` method.

        Notes
        -----
        This function assumes that the rating curve has been fitted prior to calling it.
        It also calculates the range of the stage series for generating the plot points
        if such data is available. Otherwise, it derives the range from the fitted
        observational data.
        """
        if self.fit_result is None:
            ValueError(
                "Rating curve is not fitted, call .fit() first."
            )  # TODO custom exception

        if self.stage_series is not None:
            min_stage = self.stage_series.min()
            max_stage = self.stage_series.max()
        else:
            min_stage = self._fit_obs_data.index.min()
            max_stage = self._fit_obs_data.index.max()

        stage_plt = np.linspace(min_stage, max_stage, num=100)
        q_plt = self.predict(stage=stage_plt)

        fig, ax = plt.subplots()

        if cross_section:
            stage_on_y = True

        if stage_on_y:
            y_plt = stage_plt
            x_plt = q_plt

            y_pts = self._fit_obs_data.index.to_numpy()
            x_pts = self._fit_obs_data["observed"]

            y_label = label_stage
            x_label = label_discharge
        else:
            y_plt = q_plt
            x_plt = stage_plt

            y_pts = self._fit_obs_data["observed"]
            x_pts = self._fit_obs_data.index.to_numpy()

            x_label = label_stage
            y_label = label_discharge

        ax.plot(x_pts, y_pts, "ko")
        ax.plot(x_plt, y_plt, "k-")

        ax.set_yscale(scale)
        ax.set_xscale(scale)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        return fig, ax

    # TODO
    # def plot_shift():
    #     ...

    # TODO
    # def compare(self, other):
    #     ...

    # TODO
    # def add_obs_point(self):
    #     ...

    @staticmethod
    def _dict_from_parameters(parameters):  # numpydoc ignore=GL08, PR01
        return parameters.valuesdict()
