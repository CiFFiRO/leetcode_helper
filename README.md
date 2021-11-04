# Data Structures
## DSU (disjoint set union, Union-Find)

- `__init__` - O(1)
- `__len__` - O(1)
- `number_elements` - O(1)
- `make` - ~O(1)
- `find` - ~O(1)
- `union` - ~O(1)

## Segment Tree

- `__init__` - O(n)
- `update` - O(log n)
- `value` - O(log n)

## Segment Tree with lazy propagation

- `__init__` - O(1)
- `change_range` - O(log N)
- `top` - O(1)

## Implicit Treap

- `__init__` - ~O(n log n)
- `__len__` - O(1)
- `__str__` - O(n)
- `__getitem__` - ~O(log n)
- `insert` - ~O(log n)
- `append` - ~O(log n)
- `pop` - ~O(log n)
- `value` - ~O(log n)

# Algorithms
- `next_permutation` - O(n)
- `gcd` - O(log max(a, b))
- `length_LIS` - O(n log n)
- `prefix_function` - O(n)
- `knuth_morris_prat` - O(|pattern| + |text|)
- `aho_corasick` - O(sum(|pattern|) + |text| + occ)
- `ukkonen` - O(n)
- `string_period` - O(n)
- `josephus_task` - O(n)
- `eratosthenes_sieve` - O(n log log n)
