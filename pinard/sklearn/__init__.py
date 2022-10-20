"""
The :mod:`pinard.sklearn` module includes various tools to extend 
or adapt sklearn features.
"""

from ._pipeline import FeatureAugmentation, SampleAugmentation
from ._utils import _validate_shuffle_split

__all__ = ["FeatureAugmentation", "_validate_shuffle_split", "SampleAugmentation"]
