<p align="center">
<img src="https://github.com/dobraczka/eche/raw/main/docs/assets/logo.png" alt="eche logo", width=200/>
</p>

<p align="center">
<a href="https://github.com/dobraczka/eche/actions/workflows/main.yml"><img alt="Actions Status" src="https://github.com/dobraczka/eche/actions/workflows/main.yml/badge.svg?branch=main"></a>
<a href='https://eche.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/eche/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://pypi.org/project/eche"/><img alt="Stable python versions" src="https://img.shields.io/pypi/pyversions/eche"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
</p>

Usage
=====
Eche provides a `ClusterHelper` class to conveniently handle entity clusters.

```python
  from eche import ClusterHelper
  ch = ClusterHelper([{"a1", "b1"}, {"a2", "b2"}])
  print(ch.clusters)
  {0: {'a1', 'b1'}, 1: {'a2', 'b2'}}
```

Add an element to a cluster

```python
  ch.add_to_cluster(0, "c1")
  print(ch.clusters)
  {0: {'a1', 'b1', 'c1'}, 1: {'a2', 'b2'}}
```

Add a new cluster

```python
  ch.add({"e2", "f1", "c3"})
  print(ch.clusters)
  {0: {'a1', 'b1', 'c1'}, 1: {'a2', 'b2'}, 2: {'f1', 'e2', 'c3'}}
```

Remove an element from a cluster

```python
  ch.remove("b1")
  print(ch.clusters)
  {0: {'a1', 'c1'}, 1: {'a2', 'b2'}, 2: {'f1', 'e2', 'c3'}}
```

The ``__contains__`` function is smartly overloaded. You can check if an entity is in the `ClusterHelper`:

```python
  "a1" in ch
  # True
```

If a cluster is present

```python
  {"c1","a1"} in ch
  # True
```

And even if a link exists or not

```python
  ("f1","e2") in ch
  # True
  ("a1","e2") in ch
  # False
```

To know the cluster id of an entity you can look it up with

```python
  print(ch.elements["a1"])
  0
```

To get members of a cluster either use

```python
  print(ch.members(0))
  {'a1', 'b1', 'c1'}
```

or simply

```python
  print(ch[0])
  {'a1', 'b1', 'c1'}
```

More functions can be found in the [Documentation](https://eche.readthedocs.io).

Installation
============
Simply use `pip` for installation:
```
pip install eche
```
