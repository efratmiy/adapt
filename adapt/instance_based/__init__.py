"""
Instance-Based Methods Module
"""

from ._kliep import KLIEP
from ._kmm import KMM
from ._tradaboost import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2
from ._wann import WANN
from ._ldm import LDM
from ._nnw import NearestNeighborsWeighting

__all__ = ["KLIEP", "KMM", "TrAdaBoost", "TrAdaBoostR2",
           "TwoStageTrAdaBoostR2", "WANN", "NearestNeighborsWeighting"]