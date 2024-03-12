import os
import pathlib
import shutil
from collections import OrderedDict

import numpy as np
import pytest
from eche import ClusterHelper, PrefixedClusterHelper

_LEFT_RIGHT_NAMES = ("left", "right")


@pytest.fixture()
def prefixed_cluster():
    prefixes = OrderedDict({"left": "l:", "right": "r:"})
    clusters = {
        0: {"l:a", "r:b", "r:c"},
        1: {"l:d", "l:e"},
        2: {"l:f", "l:g", "r:h", "r:i"},
    }
    return prefixes, clusters


@pytest.fixture()
def multi_source_prefixed_cluster():
    prefixes = OrderedDict({"left": "l:", "middle": "m:", "right": "r:"})
    clusters = {
        0: {"l:a", "r:b", "r:c", "m:a"},
        1: {"l:d", "l:e", "m:b"},
        2: {"l:f", "l:g", "r:h", "r:i"},
    }
    return prefixes, clusters


@pytest.fixture()
def expected_prefixed_pairs():
    return {
        ("l:a", "r:c"),
        ("l:a", "r:b"),
        ("l:g", "r:h"),
        ("l:g", "r:i"),
        ("l:f", "r:h"),
        ("l:f", "r:i"),
    }


@pytest.fixture()
def expected_prefixed_pairs_no_intra_full(expected_prefixed_pairs):
    return {
        *expected_prefixed_pairs,
        ("l:a", "m:a"),
        ("m:a", "r:b"),
        ("m:a", "r:c"),
        ("l:d", "m:b"),
        ("l:e", "m:b"),
    }


@pytest.fixture()
def expected_prefixed_pairs_intra():
    return {
        frozenset({"l:d", "l:e"}),
        frozenset({"l:f", "l:g"}),
    }


def test_clusters_init():
    clusters_1 = ClusterHelper([{"a1", "1"}, {"a2", "2"}, {"a3", "3"}])

    assert clusters_1.clusters == {0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}}
    clusters_2 = ClusterHelper({"a1": "1", "a2": "2", "a3": "3"})
    clusters_3 = ClusterHelper({0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}})
    assert clusters_1 == clusters_2
    assert clusters_1 == clusters_3

    assert clusters_1.clusters == {0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}}

    # multiple
    list_set = [{"a1", "b1", "b5"}, {"a2", "b2"}, {"a3", "b3"}]
    cluster_from_list_set_mult = ClusterHelper(list_set)
    assert cluster_from_list_set_mult.links("a1") == {"b1", "b5"}

    # init with ints
    clusters_1 = ClusterHelper([{1, 4}, {2, 5}, {3, 6}])

    assert clusters_1.clusters == {0: {1, 4}, 1: {2, 5}, 2: {3, 6}}

    # overlapping sets
    ch_sets = ClusterHelper(
        [
            {"1", "b3", "a1"},
            {"1", "b1"},
            {"b3", "a1", "b1"},
            {"a1", "b1"},
            {"c1", "e1"},
            {"c1", "d1"},
            {"e1", "d1"},
            {"a2", "2"},
        ]
    )
    expected_clusters = {
        frozenset({"a1", "1", "b1", "b3"}),
        frozenset({"a2", "2"}),
        frozenset({"c1", "d1", "e1"}),
    }
    assert {frozenset(c) for c in ch_sets.clusters.values()} == expected_clusters

    # assert no multiple cluster memberships with cluster init
    with pytest.raises(ValueError, match="multiple membership"):
        ClusterHelper({0: {"1", "2"}, 1: {"1", "3"}})


def test_cluster_links():
    clusters = ClusterHelper([{"a1", "1"}, {"a2", "2"}, {"a3", "3"}])
    assert clusters.links("a1") == "1"
    assert clusters.links("1") == "a1"
    assert clusters.links("a1", always_return_set=True) == {"1"}
    with pytest.raises(KeyError):
        clusters.links("wrong")

    clusters_1 = ClusterHelper([{1, 4}, {2, 5}, {3, 6}])
    assert clusters_1.links(1) == 4  # noqa: PLR2004
    assert clusters_1.links(4) == 1
    assert clusters_1.links(1, always_return_set=True) == {4}
    with pytest.raises(KeyError):
        clusters_1.links("wrong")


def test_cluster_members():
    clusters = ClusterHelper([{"a1", "1"}, {"a2", "2"}, {"a3", "3"}])
    assert clusters.members(clusters.elements["a1"]) == {"a1", "1"}

    clusters_1 = ClusterHelper([{1, 4}, {2, 5}, {3, 6}])
    assert clusters_1.members(clusters_1.elements[1]) == {1, 4}


def test_cluster_element_add():
    clusters_1 = ClusterHelper([{"a2", "2"}, {"a3", "3"}])
    clusters_1.add({"a1", "1", "d"})

    assert clusters_1.links("a1") == {"1", "d"}


def test_add_to_cluster():
    clusters_1 = ClusterHelper([{1, 4}, {2, 5}, {3, 6}])
    clusters_1.add_to_cluster(clusters_1.elements[1], "d")
    assert clusters_1.links(1) == {4, "d"}
    clusters_2 = ClusterHelper([{2, 5}, {3, 6}])
    clusters_2.add({1, 4, "d"})

    assert clusters_1.links(1) == clusters_2.links(1)

    # no merging
    with pytest.raises(ValueError, match="already belongs"):
        clusters_1.add_to_cluster(clusters_1.elements[2], 1)


def test_merge():
    cluster = ClusterHelper({0: {1, 2}, 1: {3, 4}})
    with pytest.raises(ValueError, match="existing"):
        cluster.merge(2, 3)
    with pytest.raises(ValueError, match="existing"):
        cluster.merge(1, 3)
    with pytest.raises(ValueError, match="existing"):
        cluster.merge(3, 1)

    cluster.merge(0, 1)
    assert cluster == ClusterHelper({0: {1, 2, 3, 4}})

    cluster = ClusterHelper({0: {1, 2}, 1: {3, 4}})
    cluster.merge(0, 1, new_id=2)
    assert cluster == ClusterHelper({2: {1, 2, 3, 4}})


def test_add():
    new_set = {1, 2}
    cluster = ClusterHelper()
    cluster.add(new_set)
    assert cluster[0] == new_set

    cid = 1
    cluster = ClusterHelper()
    cluster.add(new_set, cid)
    assert cluster[cid] == new_set
    with pytest.raises(TypeError, match="Expected set"):
        cluster.add([1, 2, 3], cid)
    with pytest.raises(ValueError, match="contains already"):
        cluster.add({1, 3})
    with pytest.raises(ValueError, match="id already"):
        cluster.add({3, 4}, cid)


def test_add_link():
    cluster = ClusterHelper({0: {1, 2}})

    # entirely new
    new_id = cluster.add_link(3, 4)
    assert cluster.elements[3] == cluster.elements[4]
    assert cluster[new_id] == {3, 4}

    # add to existing
    new_id = cluster.add_link(3, 5)
    assert cluster.elements[3] == cluster.elements[5]
    assert cluster[new_id] == {3, 4, 5}

    # merge
    new_id = cluster.add_link(1, 3)
    assert cluster == ClusterHelper({0: {1, 2, 3, 4, 5}})
    assert cluster[new_id] == {1, 2, 3, 4, 5}


def test_get():
    clusters = ClusterHelper({0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}})
    assert clusters.get(0) == {"a1", "1"}
    assert clusters.get(0, value={}) == {"a1", "1"}
    assert clusters.get("a1") is None
    assert clusters.get("a1", value=-1) == -1

    assert clusters.get(3) is None
    assert clusters.get(3, value={}) == {}


def test_cluster_element_remove():
    clusters_1 = ClusterHelper([{"a1", "1"}, {"a2", "2"}, {"a3", "3"}])
    clusters_1.remove("a1")

    with pytest.raises(KeyError):
        clusters_1.links("a1")

    with pytest.raises(KeyError):
        clusters_1.links("1")

    clusters_2 = ClusterHelper([{"a1", "1", "5"}, {"a2", "2"}, {"a3", "3"}])
    clusters_2.remove("a1")

    with pytest.raises(KeyError):
        clusters_2.links("a1")

    assert clusters_2.links("1") == "5"


def test_cluster_removal():
    ch = ClusterHelper([{"a1", "b1", "b5"}, {"a2", "b2"}, {"a3", "b3"}])
    ch.remove_cluster(ch.elements["a1"])
    assert "a1" not in ch
    assert "b1" not in ch
    assert "b5" not in ch


def test_sample():
    clusters = {0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}}
    ch = ClusterHelper(clusters)
    samp = ch.sample(1)
    c_id = next(iter(samp.clusters.keys()))
    assert len(samp) == 1
    assert c_id in clusters
    assert samp[c_id] == clusters[c_id]


def test_number_of_links():
    clusters = {0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}}
    ch = ClusterHelper(clusters)
    assert ch.number_of_links == 3  # noqa: PLR2004
    ch.add({"a4", "4"})
    assert ch.number_of_links == 4  # noqa: PLR2004
    ch.add_to_cluster(0, "a5")
    ch.add_to_cluster(0, "a6")
    assert ch.number_of_links == 9  # noqa: PLR2004


def assert_equal_pairs(actual, desired):
    actual = list(actual)
    assert len(actual) == len(desired)
    actual_set = {frozenset(p) for p in actual}
    desired_set = {frozenset(p) for p in desired}
    assert actual_set == desired_set


def test_get_all_pairs():
    clusters = {0: {"a1", "1"}, 1: {"a2", "2"}, 2: {"a3", "3"}}
    ch = ClusterHelper(clusters)
    all_pairs = ch.all_pairs()
    assert_equal_pairs(all_pairs, [("a1", "1"), ("a2", "2"), ("a3", "3")])

    all_pairs = ch.all_pairs(0)
    assert_equal_pairs(all_pairs, [("a1", "1")])

    clusters = {0: {"a1", "1", "b1", "b3"}, 1: {"a2", "2"}, 2: {"a3", "3"}}
    ch = ClusterHelper(clusters)
    all_pairs = ch.all_pairs()
    assert_equal_pairs(
        all_pairs,
        [
            ("a1", "1"),
            ("a1", "b1"),
            ("a1", "b3"),
            ("1", "b1"),
            ("1", "b3"),
            ("b1", "b3"),
            ("a2", "2"),
            ("a3", "3"),
        ],
    )

    all_pairs = ch.all_pairs(0)
    assert_equal_pairs(
        all_pairs,
        [
            ("a1", "1"),
            ("a1", "b1"),
            ("a1", "b3"),
            ("1", "b1"),
            ("1", "b3"),
            ("b1", "b3"),
        ],
    )


def test_contains():
    ch = ClusterHelper({0: {"a1", "1", "b1", "b3"}, 1: {2, 3}, 2: {"a3", "3"}})
    assert "1" in ch
    assert "5" not in ch
    assert 5 not in ch  # noqa: PLR2004
    assert ("a1", "1") in ch
    assert {"a1", "1", "b1", "b3"} in ch
    assert 2 in ch  # noqa: PLR2004
    assert (2, 3) in ch
    assert {2, 3} in ch


def test_clone():
    ch = ClusterHelper({0: {"a1", "1", "b1", "b3"}, 1: {2, 3}, 2: {"a3", "3"}})
    cloned = ch.clone()
    assert ch == cloned
    cloned.remove(2)
    assert ch != cloned


@pytest.mark.parametrize(("write_cid", "read_cid"), [(False, False), (True, True)])
def test_from_to_file(write_cid, read_cid, tmp_path):
    file_path = tmp_path / "cluster_file"
    ch = ClusterHelper({0: {"a", "b", "c"}, 1: {"d", "e"}, 2: {"f", "g", "h", "i"}})
    ch.to_file(file_path, write_cluster_id=write_cid)
    file_ch = ClusterHelper.from_file(file_path, has_cluster_id=read_cid)
    assert file_ch == ch


def test_prefixed_cluster(prefixed_cluster):
    prefixes, clusters = prefixed_cluster
    with pytest.raises(ValueError, match="known prefix"):
        PrefixedClusterHelper(ds_prefixes=prefixes, data=[{"l:a", "b"}])
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    assert len(ch.known_prefixes) == len(ch.ds_names)


def test_prefixed_cluster_add(prefixed_cluster):
    prefixes, clusters = prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    with pytest.raises(ValueError, match="known prefix"):
        ch.add({"l:abc", "lorem"})
    new_pair = {"l:abc", "r:orem"}
    ch.add(new_pair)
    assert tuple(new_pair) in ch


def test_prefixed_cluster_add_link(prefixed_cluster):
    prefixes, clusters = prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    with pytest.raises(ValueError, match="known prefix"):
        ch.add_link("l:abc", "lorem")
    new_pair = ("l:abc", "r:orem")
    ch.add_link(e1=new_pair[0], e2=new_pair[1])
    assert new_pair in ch


def test_prefixed_cluster_add_to_cluster(prefixed_cluster):
    prefixes, clusters = prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    c_id = 0
    with pytest.raises(ValueError, match="known prefix"):
        ch.add_to_cluster(c_id, "lorem")
    correct_entity = "r:orem"
    ch.add_to_cluster(c_id=c_id, new_entity=correct_entity)
    assert ch.elements[correct_entity] == c_id


def test_pairs_in_ds_tuple_binary(prefixed_cluster, expected_prefixed_pairs):
    prefixes, clusters = prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    assert expected_prefixed_pairs == set(
        ch.pairs_in_ds_tuple(ds_tuple=_LEFT_RIGHT_NAMES)
    )


def test_pairs_in_ds_tuple_multi(
    multi_source_prefixed_cluster, expected_prefixed_pairs
):
    prefixes, clusters = multi_source_prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    assert expected_prefixed_pairs == set(
        ch.pairs_in_ds_tuple(ds_tuple=_LEFT_RIGHT_NAMES)
    )


def test_get_ds_entities(multi_source_prefixed_cluster):
    expected = {"m:a", "m:b"}
    prefixes, clusters = multi_source_prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    assert expected == set(ch.get_ds_entities("middle"))


def test_intra_dataset_pairs(
    multi_source_prefixed_cluster, expected_prefixed_pairs_intra
):
    prefixes, clusters = multi_source_prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    pair_list = list(ch.intra_dataset_pairs(ds_name="left"))
    assert len(expected_prefixed_pairs_intra) == len(pair_list)
    # tuple order does not matter in intra-dataset links
    frzset_pairs = set(map(frozenset, pair_list))
    assert expected_prefixed_pairs_intra == frzset_pairs


def test_all_pairs_no_intra(
    multi_source_prefixed_cluster, expected_prefixed_pairs_no_intra_full
):
    prefixes, clusters = multi_source_prefixed_cluster
    ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)
    assert expected_prefixed_pairs_no_intra_full == set(ch.all_pairs_no_intra())
    assert len(expected_prefixed_pairs_no_intra_full) == ch.number_of_no_intra_links


def test_from_to_numpy(multi_source_prefixed_cluster, expected_prefixed_pairs):
    prefixes, _ = multi_source_prefixed_cluster
    ch = PrefixedClusterHelper.from_numpy(
        np.array(list(expected_prefixed_pairs)), ds_prefixes=prefixes
    )
    assert expected_prefixed_pairs == {
        tuple(sorted(pair)) for pair in ch.pairs_in_ds_tuple(_LEFT_RIGHT_NAMES)
    }
    assert (
        ch.clusters
        == PrefixedClusterHelper.from_numpy(
            ch.to_numpy(), ds_prefixes=prefixes
        ).clusters
    )
    with pytest.raises(ValueError, match="binary"):
        ClusterHelper.from_numpy(np.array([["bb", "bba", "aas"]]))


def test_empty_ds_prefixes():
    with pytest.raises(ValueError, match="ds_prefixes"):
        PrefixedClusterHelper()


def _create_zipped_ent_links(
    dir_path: pathlib.Path,
    inner_path: str,
    output_filename: str,
    multi_source_prefixed_cluster,
):
    full_path = dir_path.joinpath(inner_path)
    os.makedirs(full_path.parent)
    prefixes, clusters = multi_source_prefixed_cluster
    PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters).to_file(full_path)
    return shutil.make_archive(str(dir_path.joinpath(output_filename)), "zip", dir_path)


def test_from_zipped_file(
    tmp_path, multi_source_prefixed_cluster, expected_prefixed_pairs
):
    prefixes, _ = multi_source_prefixed_cluster
    zip_name = "ds"
    inner_path = pathlib.PurePosixPath("ds_name", "inner", "ent_links")
    zip_path = _create_zipped_ent_links(
        tmp_path,
        inner_path,
        zip_name,
        multi_source_prefixed_cluster,
    )

    ch = PrefixedClusterHelper.from_zipped_file(
        path=zip_path,
        inner_path=str(inner_path),
        has_cluster_id=False,
        ds_prefixes=prefixes,
    )
    assert expected_prefixed_pairs == set(
        ch.pairs_in_ds_tuple(ds_tuple=_LEFT_RIGHT_NAMES)
    )


def test_transitivity_for_all_inits():
    gold = {0: {"a", "b", "c"}}
    assert ClusterHelper(gold).clusters == gold
    assert ClusterHelper({"a": "b", "b": "c"}).clusters == gold
    assert ClusterHelper([{"a", "b"}, {"b", "c"}]).clusters == gold
    assert ClusterHelper([{"a", "b", "c"}]).clusters == gold

    # self-links should not matter anymore
    assert ClusterHelper({"a": "a", "b": "a", "c": "b"}).clusters == gold
