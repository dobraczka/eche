from eche.graph_based_clustering import to_chain

POSSIBILITIES = {
    frozenset({frozenset({"c", "a"}), frozenset({"c", "b"})}),
    frozenset({frozenset({"c", "a"}), frozenset({"a", "b"})}),
    frozenset({frozenset({"c", "b"}), frozenset({"a", "b"})}),
}


def test_to_chain():
    assert frozenset(map(frozenset, to_chain(["a", "b", "c"]))) in POSSIBILITIES
    assert list(to_chain([])) == []
