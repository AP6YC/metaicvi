"""
metaicvi - Python library for the MetaICVI unsupervised learning/clustering metric.
"""

__version__ = "0.0.1"

# from .lib import *

import param
import math
import tqdm
import sklearn
from pathlib import Path


class MetaICVIOpts(param.Parameterized):
    """MetaICVI options parameter container.

    This param.Parameterized object contains
    """

    # classifier_selection
    # classifier_opts
    icvi_window = param.Integer(
        5,
        bounds=(1, math.inf),
        doc="The size of the sliding window of ICVI samples for computing the correlations."
    )
    correlation_window = param.Integer(
        5,
        bounds=(1, math.inf),
        doc="The size of the sliding window of correlations for computing the rocket features."
    )
    n_rocket = param.Integer(
        10,
        bounds=(1, math.inf),
        doc="Number of rocket kernels used for computing features."
    )
    rocket_file = param.Path(
        Path().absolute(),
        doc="Path to the rocket kernels file."
    )
    classifier_file = param.Path(
        Path().absolute(),
        doc="Path to the stateful classifier file."
    )
    display = param.Boolean(
        False,
        doc="A flag for displaying progress with a progressbar."
    )
    fail_on_missing = param.Boolean(
        True,
        doc="A flag to trigger program failure if a filename is provided but it does not exist."
    )
