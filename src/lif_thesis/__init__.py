"""
lif_thesis

Machine learning pipeline for real-time identification of bacterial
bioaerosols using Rapid-E fluorescence, lifetime, and scattering data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lif_thesis")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Christopher Perry"
__email__ = "chrisperry437@gmail.com"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]