# -*- coding: utf-8 -*-
"""
Grades for quality
"""
from enum import Enum

class Grades(Enum):
    BAD = -1
    UNDEF = 0
    POOR = 1
    GOOD = 2
    EXCELLENT = 3
    
    