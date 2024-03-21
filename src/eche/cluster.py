"""Methods to deal with entity clusters."""
import csv
import os
import random
import zipfile
from copy import deepcopy
from itertools import chain, combinations
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import numpy as np

from .graph_based_clustering import connected_components

_BINARY_MATCH_LEN = 2
_EMPTY_DEFAULT_DS_PREF: OrderedDict[str, str] = OrderedDict()


class ClusterHelper:
    """Convenience class for entity clusters.

    The :class:`ClusterHelper` class holds a dict mapping entities to the respective cluster_id
    and a dict with cluster_id mapping to entity sets.
    The :meth:`.add()` and :meth:`.remove()` keep the respective dicts in sync.

    Examples:
        >>> from eche import ClusterHelper
        >>> ch = ClusterHelper([{"a1", "b1"}, {"a2", "b2"}])
        >>> print(ch.clusters)
        {0: {'a1', 'b1'}, 1: {'a2', 'b2'}}

        Add an element to a cluster

        >>> ch.add_to_cluster(0, "c1")
        >>> print(ch.clusters)
        {0: {'a1', 'b1', 'c1'}, 1: {'a2', 'b2'}}

        Add a new cluster

        >>> ch.add({"e2", "f1", "c3"})
        >>> print(ch.clusters)
        {0: {'a1', 'b1', 'c1'}, 1: {'a2', 'b2'}, 2: {'f1', 'e2', 'c3'}}

        Remove an element from a cluster

        >>> ch.remove("b1")
        >>> print(ch.clusters)
        {0: {'a1', 'c1'}, 1: {'a2', 'b2'}, 2: {'f1', 'e2', 'c3'}}

        Transitively connected clusters are automatically resolved

        >>> ch = ClusterHelper([{"a", "b"}, {"b", "c"}])
        >>> print(ch.clusters)
        {0: {'a', 'b', 'c'}}

        The __contains__ function is smartly overloaded. You can
        check if an entity is in the ClusterHelper

        >>> "a1" in ch
        True

        If a cluster is present

        >>> {"c1","a1"} in ch
        True

        And even if a link exists or not

        >>> ("f1","e2") in ch
        True
        >>> ("a1","e2") in ch
        False

        To know the cluster id of an entity you can look it up with

        >>> ch.elements["a1"]
        0

        To get members of a cluster either use

        >>> ch.members(0)
        {'a1', 'b1', 'c1'}

        or simply

        >>> ch[0]
        {'a1', 'b1', 'c1'}
    """

    def _contains_overlaps(self, data):
        if len(data) > 1 and len(list(chain(*data))) > len(set.union(*data)):
            return True
        return False

    def _from_sets(self, data: Iterable[Set]):
        # merge overlapping
        data = connected_components(data)

        for cluster_id, inner in enumerate(data):
            if not isinstance(inner, set):
                raise TypeError(
                    f'Only set is allowed as list element, but got "{type(inner)}"'
                )
            inner_set = set()
            for inner_element in inner:
                self.elements[inner_element] = cluster_id
                inner_set.add(inner_element)
            self.clusters[cluster_id] = inner_set

    def _from_dict(self, data: Dict):
        for cluster_id, dict_items in enumerate(data.items()):
            left, right = dict_items
            if left == right:
                raise ValueError(f'No selflinks allowed: "{left}" -> "{right}"')
            self.elements[left] = cluster_id
            self.clusters[cluster_id] = {left}
            self.elements[right] = cluster_id
            self.clusters[cluster_id].add(right)

    def _from_clusters(self, data: Dict):
        if self._contains_overlaps(data.values()):
            raise ValueError(
                "Entries with multiple memberships are not allowed, when specifying"
                " clusters and ids explicitly"
            )
        self.elements = {
            e_id: cluster_id for cluster_id, cluster in data.items() for e_id in cluster
        }
        self.clusters = data

    def __init__(
        self,
        data: Optional[Union[Iterable[Set], Dict]] = None,
    ):
        """Initialize a ClusterHelper object with clusters.

        Args:
            data : Clusters either as list of sets, or dict with links as key, value pairs, or dict with cluster id and set of members

        Raises:
            TypeError: if data is not dict or list
            ValueError: For dict[cluster_id,member_set] if overlaps between clusters

        Note:
            Will try to merge clusters transitively if necessary.
        """
        self.elements = {}
        self.clusters = {}
        if data is None:
            return
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Only list or dict allowed, but got {type(data)}")
        if isinstance(data, dict) and len(data) != 0:
            if isinstance(next(iter(data.values())), set):
                self._from_clusters(data)
                return
            # assume binary links as key-value pairs
            # transform to iterable sets of 2
            data = (set(pair) for pair in data.items())
        if isinstance(data, Iterable):
            self._from_sets(data)

    def __repr__(self):
        return f"ClusterHelper(elements={self.elements!s},clusters={self.clusters!s})"

    def __str__(self):
        return str(self.clusters)

    def info(self) -> str:
        """Return general information about this object.

        Returns:
            information about number of entities and clusters
        """
        num_elements = len(self.elements)
        num_clusters = len(self.clusters)
        return (
            self.__class__.__name__
            + f"(# elements:{num_elements}, # clusters:{num_clusters})"
        )

    def links(self, key: str, always_return_set: bool = False) -> Union[str, Set[str]]:
        """Get entities linked to this entity.

        Args:
            key: entity id
            always_return_set: If True, return set even if only one entity is contained

        Returns:
            Either the id of the single linked entity or a set of
            ids if there is more than one link
            If always_return_set is True, will always return set
        """
        cluster = self.clusters[self.elements[key]]
        other_members = cluster.difference({key})
        if not always_return_set and len(other_members) == 1:
            return next(iter(other_members))
        return other_members

    def all_pairs(self, key=None) -> Iterable[Tuple[Any, Any]]:
        """Get all entity pairs of a specific cluster or of all clusters.

        Args:
            key: cluster id, or if None, provides pairs of all clusters.

        Returns:
            Generator that produces the wanted pairs.
        """
        if key is not None:
            return combinations(self.clusters[key], 2)
        # get pair combinations of clusters and chain generators
        return chain(*[combinations(cluster, 2) for cluster in self.clusters.values()])

    def members(self, key) -> Set:
        """Get members of a cluster.

        Args:
            key: cluster id

        Returns:
            cluster members
        """
        return self.clusters[key]

    def __getitem__(self, key):
        """Get cluster members.

        Args:
            key: cluster id

        Returns:
            Cluster members
        """
        return self.clusters[key]

    def get(self, key, value=None):
        """Return cluster's element or default value.

        Tries to return the cluster with the cluster id == key.
        If None is found return provided value.

        Args:
            key: Searched cluster id.
            value: Default value to return in case id is not present.

        Returns:
            Cluster with provided cluster_id.
        """
        try:
            return self[key]
        except KeyError:
            return value

    def __contains__(self, key) -> bool:
        """Check if entities/links/clusters are contained.

        Args:
            key: Either entity id, Tuple with two entities to check for a link between entities, or a clusters as set of entity ids
        """
        if isinstance(key, Set):
            return key in self.clusters.values()
        if (
            isinstance(key, tuple)
            and len(key) == _BINARY_MATCH_LEN
            and key[0] in self.elements
            and key[1] in self.elements
        ):
            return self.elements[key[0]] == self.elements[key[1]]
        return key in self.elements

    def __setitem__(self, key, value):
        """Not Implemented."""
        raise TypeError(
            "'ClusterHelper' does not support item assignment use .add() or .remove()"
        )

    def sample(self, n: int, seed: Optional[int] = None):
        """Sample n clusters.

        Args:
            n: Number of clusters to return.
            seed: Seed for randomness

        Returns:
           ClusterHelper with n clusters.
        """
        r_gen = random.Random(seed)
        return ClusterHelper(dict(r_gen.sample(list(self.clusters.items()), n)))

    def merge(self, c1: int, c2: int, new_id: Optional[int] = None) -> int:
        """Merge two clusters.

        Args:
            c1: Id of one cluster to merge
            c2: Id of other cluster to merge
            new_id: New id of cluster, if None take c1

        Returns:
            cluster id of merged cluster

        Raises:
            ValueError: If cluster id(s) do not exist
        """
        if c1 not in self.clusters or c2 not in self.clusters:
            raise ValueError("Can only merge on existing cluster ids")
        cluster1 = self[c1]
        cluster2 = self[c2]
        if new_id:
            del self.clusters[c1]
            for e1 in cluster1:
                del self.elements[e1]
            self.add(cluster1, c_id=new_id)
        else:
            new_id = c1
        del self.clusters[c2]
        for e2 in cluster2:
            del self.elements[e2]
            self.add_to_cluster(new_id, e2)
        return new_id

    def add_link(self, e1: str, e2: str) -> Union[int, bool]:
        """Add a new entity link.

        Either adds a link to an existing entity or
        creates a new cluster with both.

        Args:
            e1: Id of one entity that will be linked
            e2: Id of other entity that will be linked

        Returns:
            Id of cluster that was created, or
            of existing cluster that was enhanced
            Returns False if link already was present
        """
        if e1 not in self and e2 not in self:
            return self.add({e1, e2})
        if e1 in self.elements and e2 in self.elements:
            if (e1, e2) in self:
                return False
            # merging
            cluster_id_1 = self.elements[e1]
            cluster_id_2 = self.elements[e2]
            return self.merge(cluster_id_1, cluster_id_2)
        if e1 in self.elements:
            cluster_id = self.elements[e1]
            new_entity = e2
        elif e2 in self.elements:
            cluster_id = self.elements[e2]
            new_entity = e1
        self.clusters[cluster_id].add(new_entity)
        self.elements[new_entity] = cluster_id
        return cluster_id

    def add_to_cluster(self, c_id: int, new_entity: str):
        """Add an entity to a cluster.

        Args:
            c_id: Cluster id where entity will be added
            new_entity: Id of new entity

        Raises:
            KeyError: If cluster id unknown
            ValueError: If entity already belongs to other cluster
        """
        if c_id not in self.clusters:
            raise KeyError("Cluster id {c_id} unknown")
        if new_entity in self.elements:
            raise ValueError(
                'Entity id "{new_entity}" already belongs to "{self.clusters[c_id]}"'
            )
        self.elements[new_entity] = c_id
        self.clusters[c_id].add(new_entity)

    def add(
        self,
        new_entry: Set,
        c_id: Optional[int] = None,
    ) -> int:
        """Add a new cluster.

        Args:
            new_entry: New cluster as set
            c_id: Cluster id that will be assigned.
           If None, the next largest cluster id will be assigned
           Assuming cluster ids are integers

        Raises:
            ValueError: If entity id already present in other cluster
                Or if new cluster id cannot be inferred automatically
                by incrementing
        """
        if not isinstance(new_entry, set):
            raise TypeError(f"Expected set, but got {type(new_entry)}")
        if not len(new_entry.intersection(self.elements.keys())) == 0:
            raise ValueError("Set contains already present entries")
        if c_id is not None and c_id in self.clusters:
            raise ValueError("Cluster id already exists")

        if c_id is None:
            if len(self.clusters) == 0:
                c_id = 0
            else:
                max_cid = max(self.clusters.keys())
                if not isinstance(max_cid, int):
                    raise ValueError(
                        "Cluster Id cannot be automatically incremented, please provide"
                        " it explicitly"
                    )
                c_id = max_cid + 1
        self.clusters[c_id] = set()
        for e in new_entry:
            self.elements[e] = c_id
            self.clusters[c_id].add(e)
        return c_id

    def remove(self, entry: str):
        """Remove an entity.

        Args:
            entry: entity to remove
        """
        cluster_id = self.elements[entry]
        del self.elements[entry]
        cluster = self.clusters[cluster_id]
        if len(cluster) == _BINARY_MATCH_LEN:
            for member in cluster:
                if member != entry:
                    del self.elements[member]
            del self.clusters[cluster_id]
        else:
            self.clusters[cluster_id].remove(entry)

    def remove_cluster(self, cluster_id):
        """Remove an entire cluster with the given cluster id.

        Args:
            cluster_id: id of the cluster to remove
        """
        cluster_elements = self.clusters[cluster_id]
        for e in iter(cluster_elements):
            del self.elements[e]
        del self.clusters[cluster_id]

    def __eq__(self, other):
        if isinstance(other, ClusterHelper):
            return (self.clusters == other.clusters) and (
                self.elements == other.elements
            )
        return False

    def clone(self) -> "ClusterHelper":
        """Create a clone of this object.

        Returns:
            cloned ClusterHelper
        """
        cloned = ClusterHelper()
        cloned.elements = deepcopy(self.elements)
        cloned.clusters = deepcopy(self.clusters)
        return cloned

    def __len__(self):
        return len(self.clusters)

    @property
    def number_of_links(self):
        """Return the total number of links."""

        def number_of_pairs_in_set(s):
            n = len(s)
            return int(n * (n - 1) / 2)

        return sum(map(number_of_pairs_in_set, self.clusters.values()))

    @classmethod
    def _create_with_cluster_id(
        cls, csv_reader, sep: str, encoding: str
    ) -> "ClusterHelper":
        e_to_cid: Dict[Union[int, str], Set[str]] = {}
        for row in csv_reader:
            c_id: Union[str, int]
            try:
                c_id = int(row[0])
            except ValueError:
                c_id = row[0]
            entries = set(row[1:])
            e_to_cid[c_id] = entries
        return cls(e_to_cid)

    @classmethod
    def _create_without_cluster_id(
        cls, csv_reader, sep: str, encoding: str
    ) -> "ClusterHelper":
        return cls([set(row) for row in csv_reader])

    @classmethod
    def _from_IO(
        cls,
        io_object: IO,
        has_cluster_id: bool = False,
        sep: str = ",",
        encoding: str = "utf8",
    ) -> "ClusterHelper":
        if isinstance(io_object, zipfile.ZipExtFile):
            reader = csv.reader(
                (line.decode(encoding) for line in io_object), delimiter=sep
            )
        else:
            reader = csv.reader(io_object, delimiter=sep)
        if has_cluster_id:
            return cls._create_with_cluster_id(reader, sep, encoding)
        return cls._create_without_cluster_id(reader, sep, encoding)

    @classmethod
    def from_file(
        cls, path: Union[str, os.PathLike], has_cluster_id: bool = False, sep=","
    ) -> "ClusterHelper":
        """Create ClusterHelper from file.

        Expects entities seperated by `sep`, with each line representing a cluster.

        Args:
            path: path to file containing entity clusters
            has_cluster_id: if True, the first entry in each line is used as cluster id
            sep: seperator

        Returns:
            ClusterHelper
        """
        with open(path) as in_file:
            return cls._from_IO(in_file, has_cluster_id=has_cluster_id, sep=sep)

    def to_file(self, path: Union[str, os.PathLike], write_cluster_id: bool = False):
        """Write clusters to file.

        Each cluster line is consists of comma-seperated entities.

        Args:
            path: Where to write the clusters.
            write_cluster_id: If True, first entry of each line is the cluster id
        """
        with open(path, "w") as out_file:
            writer = csv.writer(out_file, delimiter=",", quotechar='"')
            for c_id, elements in self.clusters.items():
                if write_cluster_id:
                    writer.writerow([c_id, *elements])
                else:
                    writer.writerow(elements)

    @classmethod
    def from_numpy(cls, numpy_links: "np.ndarray") -> "ClusterHelper":
        """Create a ClusterHelper from a 2D numpy array (i.e. binary entity links).

        Args:
            numpy_links: binary links in 2D numpy array

        Returns:
            "ClusterHelper":

        Raises:
            ValueError if shape is not 2D
        """
        if not numpy_links.shape[1] == _BINARY_MATCH_LEN:
            raise ValueError(
                f"Can only handle binary links (i.e. 2d array) as input, but got array of shape {numpy_links.shape}"
            )
        return cls(list(map(set, numpy_links)))

    def to_numpy(self) -> "np.ndarray":
        """Return binary links as numpy array.

        Returns:
            binary links
        """
        import numpy as np

        return np.array(list(self.all_pairs()))

    @classmethod
    def from_zipped_file(
        cls,
        path: Union[str, os.PathLike],
        inner_path: str,
        has_cluster_id: bool = False,
        sep: str = ",",
        encoding: str = "utf8",
    ) -> "ClusterHelper":
        """Read an inner link file from a zip archive.

        Args:
            path: The path to the zip archive
            inner_path: The path inside the zip archive to the link file
            has_cluster_id: if True, the first entry in each line is used as cluster id
            sep: seperator
            encoding: used to decode the data

        Returns:
            ClusterHelper
        """
        with zipfile.ZipFile(file=path) as zip_file, zip_file.open(inner_path) as file:
            return cls._from_IO(
                file, has_cluster_id=has_cluster_id, sep=sep, encoding=encoding
            )


class PrefixedClusterHelper(ClusterHelper):
    """ClusterHelper which uses prefixes, to associate entities with datasets.

    Examples:
        >>> from eche import PrefixedClusterHelper
        >>> prefixes = OrderedDict({"ds1": "foo:", "ds2": "bar:", "ds3": "baz:"})
        >>> clusters = {
        ...     0: {"foo:a", "bar:b", "bar:c", "baz:a"},
        ...     1: {"foo:d", "foo:e", "baz:b"},
        ...     2: {"foo:f", "foo:g", "bar:h", "bar:i"},
        ... }
        >>> ch = PrefixedClusterHelper(ds_prefixes=prefixes, data=clusters)

        Only entities with known prefix can be added

        >>> ch.add_to_cluster(0, "unknown:b")
        ValueError: "unknown:b" does not start with any known prefix: ['foo:', 'bar:', 'baz:']

        You can ask for entity pairs of a binary combination of datasets
        The tuples are ordered as the supplied dataset tuples implies
        i.e. all the first tuple entries are entities of the first dataset

        >>> list(ch.pairs_in_ds_tuple(("ds1","ds3")))
        [('foo:a', 'baz:a'), ('foo:d', 'baz:b'), ('foo:e', 'baz:b')]

        Intradataset links can also be returned

        >>> list(ch.intra_dataset_pairs("ds2"))
        [('bar:c', 'bar:b'), ('bar:i', 'bar:h')]

        To get all entities of a single dataset you can use

        >>> list(ch.get_ds_entities("ds1"))
        ["foo:a", "foo:d", "foo:e", "foo:g", "foo:f"]
    """

    def __init__(
        self,
        data: Optional[Union[Iterable[Set], Dict]] = None,
        ds_prefixes: OrderedDict[str, str] = _EMPTY_DEFAULT_DS_PREF,
    ):
        if len(ds_prefixes) == 0:
            raise ValueError("Need ds_prefixes!")
        self.ds_prefixes = ds_prefixes
        super().__init__(data)
        for e_id in self.elements:
            self._check_known_prefix(e_id)

    def _check_known_prefix(self, eid: str):
        for pref in self.known_prefixes:
            if eid.startswith(pref):
                return
        raise ValueError(
            f'"{eid}" does not start with any known prefix: {self.known_prefixes}'
        )

    @property
    def ds_names(self) -> List[str]:
        return list(self.ds_prefixes.keys())

    @property
    def known_prefixes(self) -> List[str]:
        return list(self.ds_prefixes.values())

    def add(
        self,
        new_entry: Set,
        c_id: Optional[int] = None,
    ) -> int:
        """Add a new cluster.

        Args:
            new_entry: New cluster as set
            c_id: Cluster id that will be assigned.
           If None, the next largest cluster id will be assigned
           Assuming cluster ids are integers

        Raises:
            ValueError: If entity id already present in other cluster
                Or if new cluster id cannot be inferred automatically by incrementing
                Or if prefix is unknown
        """
        for e_id in new_entry:
            self._check_known_prefix(e_id)
        return super().add(new_entry=new_entry, c_id=c_id)

    def add_link(self, e1: str, e2: str) -> Union[int, bool]:
        """Add a new entity link.

        Either adds a link to an existing entity or
        creates a new cluster with both.

        Args:
            e1: Id of one entity that will be linked
            e2: Id of other entity that will be linked

        Returns:
            Id of cluster that was created, or
            of existing cluster that was enhanced
            Returns False if link already was present

        Raises:
            ValueError: If prefix is unknown
        """
        self._check_known_prefix(e1)
        self._check_known_prefix(e2)
        return super().add_link(e1=e1, e2=e2)

    def add_to_cluster(self, c_id: int, new_entity: str):
        """Add an entity to a cluster.

        Args:
            c_id: Cluster id where entity will be added
            new_entity: Id of new entity

        Raises:
            KeyError: If cluster id or prefix is unknown
            ValueError: If entity already belongs to other cluster
        """
        self._check_known_prefix(new_entity)
        return super().add_to_cluster(c_id=c_id, new_entity=new_entity)

    def get_ds_entities(self, ds_name: str) -> Generator[str, None, None]:
        """Get all entities belonging to the given dataset.

        Args:
            ds_name: Name of dataset

        Returns:
            Generator producing entity ids of the given dataset
        """
        prefix = self.ds_prefixes[ds_name]
        for ele in self.elements:
            if ele.startswith(prefix):
                yield ele

    def pairs_in_ds_tuple(
        self, ds_tuple: Tuple[str, str]
    ) -> Generator[Tuple[str, str], None, None]:
        """Returns known links between two given datasets.

        Args:
            ds_tuple: Dataset tuple

        Returns:
            Generator that produces known links between two given datasets.
        """
        for ds_name in ds_tuple:
            if ds_name not in self.ds_names:
                raise ValueError(f"Unknown dataset name {ds_name}")
        if len(ds_tuple) != _BINARY_MATCH_LEN:
            raise ValueError(f"Expected tuple of length 2, but got {len(ds_tuple)}!")
        lpref = self.ds_prefixes[ds_tuple[0]]
        rpref = self.ds_prefixes[ds_tuple[1]]
        for pair in super().all_pairs():
            left, right = None, None
            if pair[0].startswith(lpref) and pair[1].startswith(rpref):
                left = pair[0]
                right = pair[1]
            elif pair[1].startswith(lpref) and pair[0].startswith(rpref):
                left = pair[1]
                right = pair[0]
            if left is not None and right is not None:
                yield left, right
            else:
                continue

    def intra_dataset_pairs(
        self, ds_name: str
    ) -> Generator[Tuple[str, str], None, None]:
        """Returns known links inside given dataset.

        Args:
            ds_name: Dataset for which to find links

        Returns:
            Generator that produces known links inside given dataset.
        """
        return self.pairs_in_ds_tuple(ds_tuple=(ds_name, ds_name))

    def all_pairs_no_intra(self) -> Generator[Tuple[str, str], None, None]:
        """Returns known links without intra-dataset pairs (with canonical ordering).

        Returns:
            Generator that produces known links without intra-dataset pairs.
        """

        def _find_prefix(e_id, prefixes) -> Tuple[str, int]:
            for idx, pref in enumerate(self.known_prefixes):
                if e_id.startswith(pref):
                    return pref, idx
            # we should never get here, because we checked this
            raise ValueError(f"Unknown prefix for {e_id}")

        for pair in super().all_pairs():
            lpref, lidx = _find_prefix(pair[0], self.known_prefixes)
            rpref, ridx = _find_prefix(pair[1], self.known_prefixes)
            if lpref != rpref:
                # check canonical order
                if lidx > ridx:
                    yield pair[1], pair[0]
                else:
                    yield pair

    @property
    def number_of_no_intra_links(self) -> int:
        """Returns number of links without intra-dataset links."""
        return sum(1 for _ in self.all_pairs_no_intra())

    @property
    def number_of_intra_links(self) -> Tuple[int, ...]:
        """Returns number of intra-dataset links for each dataset."""
        return tuple(
            sum(1 for _ in self.intra_dataset_pairs(ds_name=ds_name))
            for ds_name in self.ds_names
        )

    @classmethod
    def from_numpy(
        cls,
        numpy_links: "np.ndarray",
        ds_prefixes: OrderedDict[str, str] = _EMPTY_DEFAULT_DS_PREF,
    ) -> "PrefixedClusterHelper":
        """Create a PrefixedClusterHelper from a 2D numpy array.

        Args:
            numpy_links: binary links in 2D numpy array
            ds_prefixes: Dataset prefixes

        Returns:
            "PrefixedClusterHelper":
        """
        if ds_prefixes is None:
            raise ValueError("Need ds_prefixes!")
        # don't use super, else it will use this cls
        ch = ClusterHelper.from_numpy(numpy_links)
        return cls(data=ch.clusters, ds_prefixes=ds_prefixes)

    @classmethod
    def from_file(
        cls,
        path: Union[str, os.PathLike],
        has_cluster_id: bool = False,
        sep=",",
        ds_prefixes: OrderedDict[str, str] = _EMPTY_DEFAULT_DS_PREF,
    ) -> "PrefixedClusterHelper":
        """Create PrefixedClusterHelper from file.

        Expects entities seperated by `sep`, with each line representing a cluster.

        Args:
            path: path to file containing entity clusters
            has_cluster_id: if True, the first entry in each line is used as cluster id
            sep: seperator
            ds_prefixes: Dataset prefixes

        Returns:
            PrefixedClusterHelper
        """
        if ds_prefixes is None:
            raise ValueError("Need ds_prefixes!")
        # don't use super, else it will use this cls
        ch = ClusterHelper.from_file(
            path=path,
            has_cluster_id=has_cluster_id,
            sep=sep,
        )
        return cls(data=ch.clusters, ds_prefixes=ds_prefixes)

    @classmethod
    def from_zipped_file(
        cls,
        path: Union[str, os.PathLike],
        inner_path: str,
        has_cluster_id: bool = False,
        sep: str = ",",
        encoding: str = "utf8",
        ds_prefixes: OrderedDict[str, str] = _EMPTY_DEFAULT_DS_PREF,
    ) -> "PrefixedClusterHelper":
        """Read an inner link file from a zip archive.

        Args:
            path: The path to the zip archive
            inner_path: The path inside the zip archive to the link file
            has_cluster_id: if True, the first entry in each line is used as cluster id
            sep: seperator
            encoding: used to decode the data
            ds_prefixes: Dataset prefixes

        Returns:
            "PrefixedClusterHelper":
        """
        if ds_prefixes is None:
            raise ValueError("Need ds_prefixes!")
        # don't use super, else it will use this cls
        ch = ClusterHelper.from_zipped_file(
            path=path,
            inner_path=inner_path,
            has_cluster_id=has_cluster_id,
            sep=sep,
            encoding=encoding,
        )
        return cls(data=ch.clusters, ds_prefixes=ds_prefixes)
