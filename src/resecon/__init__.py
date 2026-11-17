"""resecon: optimal extraction of a renewable resource by value function iteration.

Example
-------
>>> import numpy as np
>>> from resecon import ResourceExtraction
>>> model = ResourceExtraction(h=lambda x: x * (1 - x), u=np.sqrt, delta=0.8,
...                            xmin=0, xmax=1).solve()
>>> model.steady_states()        # doctest: +SKIP
array([0.37487487])
"""

from importlib.metadata import PackageNotFoundError, version

from .model import ResourceExtraction

try:
    __version__ = version("resecon")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = ["ResourceExtraction", "__version__"]
