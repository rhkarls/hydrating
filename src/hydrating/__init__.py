# -*- coding: utf-8 -*-
import importlib.metadata

from . import models as models
from .core.hydrating import RatingCurve as RatingCurve

__version__ = importlib.metadata.version("hydrating")
