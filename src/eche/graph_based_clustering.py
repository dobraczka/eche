from typing import Collection, Generator, Iterable, Set, Tuple, TypeVar

import networkx as nx
from networkx.algorithms.components.connected import connected_components as nx_con_com

T = TypeVar("T")


def to_graph(elements: Iterable[Collection]) -> nx.Graph:
    """Create chain graph from iterable of collections.

    Args:
        elements: Iterable of collections
    """
    G = nx.Graph()
    for part in elements:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a chain of edges:
        G.add_edges_from(to_chain(part))
    return G


def to_chain(elements: Collection[T]) -> Generator[Tuple[T, T], None, None]:
    """Return elements as chain of edges.

    Args:
        elements: Container with elements

    Examples:
        >>> to_chain(['a','b','c','d'])
        [(a,b), (b,c),(c,d)]
    """
    it = iter(set(elements))
    last = next(it)

    for current in it:
        yield last, current
        last = current


def connected_components(
    clusters: Iterable[Collection[T]],
) -> Generator[Set[T], None, None]:
    """Create connected components from iterable of known clusters.

    Args:
        clusters: Known clusters

    Returns:
        Generator of connected components
    """
    # https://stackoverflow.com/a/4843408
    G = to_graph(clusters)
    return nx_con_com(G)
