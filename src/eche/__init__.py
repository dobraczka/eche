from importlib.metadata import version  # pragma: no cover

from .cluster import ClusterHelper, PrefixedClusterHelper
from .graph_based_clustering import connected_components

__version__ = version(__package__)

__all__ = ["ClusterHelper", "PrefixedClusterHelper", "connected_components"]
