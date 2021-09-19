import math
import random
import bisect
import collections
from typing import Any, Optional, Callable


class DSU:
    """Union-Find, disjoint-set"""
    def __init__(self, is_equal_compare: bool = True) -> None:
        """Create object. O(1)

        :param is_equal_compare: compare elements by '=='. False if compare by 'is'.
        """
        self.to_parent = dict()
        self.is_equal_compare = is_equal_compare
        self.size = 0
        self.elements_number = 0

    def number_elements(self) -> int:
        """Return number elements in all sets. O(1)

        :return:
        """
        return self.elements_number

    def __len__(self) -> int:
        """Return number of sets. O(1)

        :return:
        """
        return self.size

    def make(self, x: Any) -> None:
        """Create new set. ~O(1)

        :param x: element.
        :return:
        """
        self.to_parent[x] = x
        self.size += 1
        self.elements_number += 1

    def find(self, x: Any) -> Optional[Any]:
        """Return mark set by element. None if element not in any set. ~O(1)

        :param x: element.
        :return:
        """
        if x not in self.to_parent:
            return None

        if (self.is_equal_compare and self.to_parent[x] == x) or (not self.is_equal_compare and self.to_parent[x] is x):
            return x
        self.to_parent[x] = self.find(self.to_parent[x])
        return self.to_parent[x]

    def union(self, x: Any, y: Any) -> None:
        """Union two sets by elements. ~O(1)

        :param x: element is first set.
        :param y: element is second set.
        :return:
        """
        x_mark = self.find(x)
        y_mark = self.find(y)

        if (self.is_equal_compare and x_mark != y_mark) or (self.is_equal_compare and x_mark is not y_mark):
            self.size -= 1

        if random.randint(0, 1):
            self.to_parent[x_mark] = y_mark
        else:
            self.to_parent[y_mark] = x_mark


class SegmentTree:
    """Segment tree, statistic tree."""
    def __init__(self, data: list[Any], monoid: Callable[[Any, Any], Any], neutral_element: Any):
        """Create tree. O(n)

        :param data: target list.
        :param monoid: associative operation.
        :param neutral_element: neutral element for this operation.
        """
        self.tree = [0] * (2 ** (math.ceil(math.log2(len(data))) + 1) - 1)
        self.monoid = monoid
        for i in range(len(self.tree) // 2 + 1, len(self.tree) // 2 + len(data)):
            self.tree[i] = neutral_element

        for i in range(len(self.tree) // 2 - 1, -1, -1):
            left, right = 2 * i + 1, 2 * i + 2
            self.tree[i] = self.monoid(self.tree[left], self.tree[right])

    def update(self, i: int, x: Any) -> None:
        """Update value by index. O(log n)

        :param i: index.
        :param x: new value.
        :return:
        """
        i += len(self.tree) // 2
        self.tree[i] = x
        parent = (i // 2 - 1) if i % 2 == 0 else i // 2
        while parent >= 0:
            self.tree[parent] = self.monoid(self.tree[2 * parent + 1], self.tree[2 * parent + 2])
            parent = (parent // 2 - 1) if parent % 2 == 0 else parent // 2

    def value(self, i: int, j: int) -> None:
        """Return aggregated value by monoid in interval [i; j]. O(log n)

        :param i: begin interval.
        :param j: end interval.
        :return:
        """
        left_res, right_res = 0, 0
        i += len(self.tree) // 2
        j += len(self.tree) // 2
        while i < j:
            if i % 2 == 0:
                left_res = self.monoid(left_res, self.tree[i])
            if j % 2 != 0:
                right_res = self.monoid(self.tree[j], right_res)
            i //= 2
            j = (j - 2) // 2
        if i == j:
            left_res = self.monoid(left_res, self.tree[i])
        return self.monoid(left_res, right_res)


def length_LIS(nums: list[Any]) -> int:
    """Return length os longest increased subsequence. O(n log n)

    :param nums: sequence.
    :return:
    """
    d = [+math.inf] * (len(nums) + 1)
    d[0] = -math.inf
    d[1] = nums[0]
    result = 1
    for i in range(2, len(nums) + 1):
        x = nums[i - 1]
        index = bisect.bisect_left(d, x)

        if index == 0:
            index += 1

        if d[index] == math.inf:
            result += 1
        d[index] = x

    return result


def gcd(a: int, b: int) -> int:
    """Return greater common divider. O(log max(a, b))

    :param a: first number.
    :param b: second number.
    :return:
    """
    while b != 0:
        b, a = a % b, b
    return a


def next_permutation(permutation: list[Any]) -> Optional[list[Any]]:
    """Return next permutation. None if next not exist. O(n)

    :param permutation:
    :return:
    """
    for i in range(len(permutation) - 2, -1, -1):
        if permutation[i] < permutation[i + 1]:
            k = i
            break
    else:
        return None

    for i in range(len(permutation) - 1, k, -1):
        if permutation[k] < permutation[i]:
            r = i
            break

    permutation = list(permutation)
    permutation[k], permutation[r] = permutation[r], permutation[k]
    return permutation[0:k + 1] + list(reversed(permutation[k + 1:]))


class ImplicitTreap:
    """List by implicit treap (implicit Cartesian tree)."""
    class ImplicitTreapNode:
        """Node treap."""
        def __init__(
                self, y: float, value: Any, left: 'ImplicitTreap.ImplicitTreapNode' = None,
                right: 'ImplicitTreap.ImplicitTreapNode' = None
        ):
            """Create object. O(1)

            :param y: priority.
            :param value: value stored in node.
            :param left: left child.
            :param right: right child.
            """
            self.y = y
            self.value = value
            self.left = left
            self.right = right
            self.size = 1
            self.aggregate_value = None

        def recalc(self, monoid: Callable = None, neutral_element: Any = None) -> None:
            """Recalculate size and aggregate value. O(1)

            :param monoid: associative operation.
            :param neutral_element: neutral element for this operation.
            :return:
            """
            def aggregate_value(node):
                if node is None:
                    return neutral_element
                if node.aggregate_value is None:
                    return node.value
                return node.aggregate_value

            self.size = 1
            self.size += self.left.size if self.left is not None else 0
            self.size += self.right.size if self.right is not None else 0
            if monoid is not None:
                self.aggregate_value = monoid(
                    monoid(
                        self.value, aggregate_value(self.left)
                    ),
                    aggregate_value(self.right)
                )

    def __init__(self, values: list[Any] = None, monoid: Callable[[Any, Any], Any] = None, neutral_element: Any = None):
        """Create list. ~O(n log n)

        :param values: initialize values.
        :param monoid: associative operation.
        :param neutral_element: neutral element for this operation.
        """
        self.root = None
        self.monoid = monoid
        self.neutral_element = neutral_element
        if values is not None:
            for value in values:
                self.append(value)

    def _merge(
            self, left: 'ImplicitTreap.ImplicitTreapNode',
            right: 'ImplicitTreap.ImplicitTreapNode'
    ) -> Optional['ImplicitTreap.ImplicitTreapNode']:
        """Merge two treaps. ~O(log n)

        :param left: first treap.
        :param right: second treap.
        :return:
        """
        if left is None:
            return right
        if right is None:
            return left

        if left.y > right.y:
            tmp = self._merge(left.right, right)
            result = ImplicitTreap.ImplicitTreapNode(left.y, left.value, left.left, tmp)
        else:
            tmp = self._merge(left, right.left)
            result = ImplicitTreap.ImplicitTreapNode(right.y, right.value, tmp, right.right)

        result.recalc(self.monoid, self.neutral_element)
        return result

    def _split(
            self, node: 'ImplicitTreap.ImplicitTreapNode', index: int
    ) -> tuple[Optional['ImplicitTreap.ImplicitTreapNode'], Optional['ImplicitTreap.ImplicitTreapNode']]:
        """Split treap by index. ~O(log n)

        :param node: treap.
        :param index: index element.
        :return:
        """
        if node is None:
            return None, None

        left_size = node.left.size if node.left is not None else 0
        if index >= left_size + 1:
            if node.right is None:
                tmp, right = None, None
            else:
                tmp, right = self._split(node.right, index - left_size - 1)

            left = ImplicitTreap.ImplicitTreapNode(node.y, node.value, node.left, tmp)
            left.recalc(self.monoid, self.neutral_element)
        else:
            if node.left is None:
                left, tmp = None, None
            else:
                left, tmp = self._split(node.left, index)

            right = ImplicitTreap.ImplicitTreapNode(node.y, node.value, tmp, node.right)
            right.recalc(self.monoid, self.neutral_element)

        return left, right

    def __getitem__(self, index: int) -> Any:
        """Return value by index in list. ~O(log n)

        :param index: index is list.
        :return:
        """
        if self.root is None or index > self.root.size or index < -self.root.size:
            raise Exception('Index out of range')
        if index < 0:
            index = -1 - index

        node = self.root
        while True:
            left_size = node.left.size if node.left is not None else 0
            if index < left_size:
                node = node.left
            elif node.left is not None and index == 0:
                return node.value
            elif index == left_size:
                return node.value
            else:
                node = node.right
                index -= left_size + 1

    def insert(self, index: int, value: Any) -> None:
        """Insert value in list by index. ~O(log n)

        :param index: index in list.
        :param value: value.
        :return:
        """
        if self.root is None:
            self.root = ImplicitTreap.ImplicitTreapNode(random.random(), value)
            return

        if index > self.root.size:
            raise Exception('Index out of range')

        left, right = self._split(self.root, index)
        node = self.ImplicitTreapNode(random.random(), value)
        self.root = self._merge(self._merge(left, node), right)

    def append(self, value: Any) -> None:
        """Insert value in tail list. ~O(log n)

        :param value: value.
        :return:
        """
        self.insert(self.root.size if self.root is not None else 0, value)

    def pop(self, index: int = None) -> Any:
        """Pop element by index from list. ~O(log n)

        :param index: index.
        :return:
        """
        if index is None:
            index = self.root.size - 1
        if self.root is None or index > self.root.size or index < -self.root.size:
            raise Exception('Index out of range')
        if index < 0:
            index = -1 - index

        left, right = self._split(self.root, index)
        result, right = self._split(right, 1)
        self.root = self._merge(left, right)
        return result.value

    def __len__(self) -> int:
        """Return length list. O(1)

        :return:
        """
        return self.root.size if self.root is not None else 0

    def __str__(self) -> str:
        """Return string representation of list. O(n)

        :return:
        """
        result = []

        def dfs(node):
            nonlocal result
            if node is None:
                return

            dfs(node.left)
            result.append(node.value)
            dfs(node.right)

        dfs(self.root)
        return str(result)

    def value(self, begin: int, end: int) -> Any:
        """Return calculated value on interval [begin; end) by monoid. ~O(log n)

        :param begin: left index.
        :param end: right index.
        :return:
        """
        _, right = self._split(self.root, begin)
        mid, _ = self._split(right, end - begin)
        return self.neutral_element if mid is None else mid.aggregate_value


def prefix_function(string: str) -> list[int]:
    """Return prefix function for string. O(n)

    :param string: input string.
    :return:
    """
    result = [0] * len(string)

    result[0] = 0
    for i in range(1, len(string)):
        k = result[i - 1]
        while k > 0 and string[i] != string[k]:
            k = result[k-1]
        if string[i] == string[k]:
            k += 1
        result[i] = k

    return result


def knuth_morris_prat(pattern, text) -> list[int]:
    """Knuth-Morris-Prat algorithm by prefix function. O(|pattern| + |text|)
    Return indexes of occurrences pattern in text.

    :param pattern: searched pattern.
    :param text: text.
    :return:
    """
    pi = prefix_function(pattern + '#' + text)
    result = []
    for i in range(len(text)):
        if pi[i + len(pattern) + 1] == len(pattern):
            result.append(i)
    return result


def aho_corasick(patterns: list[str], text: str):
    """Aho-Corasick algorithm. O(sum(|pattern|) + |text| + occ)
    Return occurrences of patterns in text.

    :param patterns: searched patterns.
    :param text: text.
    :return:
    """
    class Node:
        node_index = 0

        def __init__(
                self, parent: Optional['Node'] = None, by_character: Optional[str] = None,
                index_pattern: Optional[int] = None
        ):
            self.parent = parent
            self.by_character = by_character
            self.index_pattern = index_pattern
            self.edges = dict()
            self.id = self.node_index
            self.node_index += 1

        def __hash__(self) -> int:
            return self.id

        def __str__(self) -> str:
            result = []

            def dfs(node):
                nonlocal result
                result.append(f'{node.index_pattern} -> ' + '{')
                for character in node.edges:
                    result.append(f'{character}: -> ' + '{')
                    dfs(node.edges[character])
                    result.append('} ')
                result.append('} ')

            dfs(self)
            return ''.join(result)

    root = Node()
    for index, pattern in enumerate(patterns):
        node = root
        for character in pattern:
            if character not in node.edges:
                new_node = Node(node, character)
                node.edges[character] = new_node
                node = new_node
            else:
                node = node.edges[character]
        node.index_pattern = index

    states_table = dict()
    suffix_links_table = dict()
    optimal_suffix_links_table = dict()

    def next_state(node: 'Node', character: str) -> 'Node':
        """Return next state in SM.

        :param node: node in trie.
        :param character:
        :return:
        """
        nonlocal states_table

        if (node, character) in states_table:
            return states_table[(node, character)]

        if character in node.edges:
            result = node.edges[character]
        elif node is root:
            result = root
        else:
            result = next_state(suffix_link(node), character)

        states_table[(node, character)] = result
        return result

    def suffix_link(node: 'Node') -> 'Node':
        """Return suffix link.

        :param node: node in trie.
        :return:
        """
        nonlocal suffix_links_table

        if node in suffix_links_table:
            return suffix_links_table[node]

        if node is root or node.parent is root:
            result = root
        else:
            result = next_state(suffix_link(node.parent), node.by_character)

        suffix_links_table[node] = result
        return result

    def optimal_suffix_link(node: 'Node') -> 'Node':
        """Return compressed suffix link.

        :param node: node in trie.
        :return:
        """
        nonlocal optimal_suffix_links_table

        if node in optimal_suffix_links_table:
            return optimal_suffix_links_table[node]

        link = suffix_link(node)
        if link.index_pattern is not None:
            result = link
        elif node is root:
            result = root
        else:
            result = optimal_suffix_link(link)

        optimal_suffix_links_table[node] = result

        return result

    result = collections.defaultdict(list)
    current_node = root
    for index, character in enumerate(text):
        current_node = next_state(current_node, character)
        node = current_node
        while node is not root:
            if node.index_pattern is not None:
                result[node.index_pattern].append(
                    index - len(patterns[node.index_pattern]) + 1
                )
            node = optimal_suffix_link(node)

    return result


