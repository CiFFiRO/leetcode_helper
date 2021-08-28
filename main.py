import math
import random
import bisect


class DSU:
    def __init__(self, is_equal_compare):
        self.to_parent = dict()
        self.is_equal_compare = is_equal_compare
        self.size = 0
        self.elements_number = 0

    def number_elements(self):
        return self.elements_number

    def __len__(self):
        return self.size

    def make(self, x):
        self.to_parent[x] = x
        self.size += 1
        self.elements_number += 1

    def find(self, x):
        if x not in self.to_parent:
            return None

        if (self.is_equal_compare and self.to_parent[x] == x) or (not self.is_equal_compare and self.to_parent[x] is x):
            return x
        self.to_parent[x] = self.find(self.to_parent[x])
        return self.to_parent[x]

    def union(self, x, y):
        x_mark = self.find(x)
        y_mark = self.find(y)

        if (self.is_equal_compare and x_mark != y_mark) or (self.is_equal_compare and x_mark is not y_mark):
            self.size -= 1

        if random.randint(0, 1):
            self.to_parent[x_mark] = y_mark
        else:
            self.to_parent[y_mark] = x_mark


class SegmentTree:
    def __init__(self, data, monoid, neutral_element):
        self.tree = [0] * (2 ** (math.ceil(math.log2(len(data))) + 1) - 1)
        self.monoid = monoid
        for i in range(len(self.tree) // 2 + 1, len(self.tree) // 2 + len(data)):
            self.tree[i] = neutral_element

        for i in range(len(self.tree) // 2 - 1, -1, -1):
            left, right = 2 * i + 1, 2 * i + 2
            self.tree[i] = self.monoid(self.tree[left], self.tree[right])

    def update(self, i, x):
        i += len(self.tree) // 2
        self.tree[i] = x
        parent = (i // 2 - 1) if i % 2 == 0 else i // 2
        while parent >= 0:
            self.tree[parent] = self.monoid(self.tree[2 * parent + 1], self.tree[2 * parent + 2])
            parent = (parent // 2 - 1) if parent % 2 == 0 else parent // 2

    def value(self, i, j):
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


def length_LIS(nums):
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


def gcd(a, b):
    while b != 0:
        b, a = a % b, b
    return a


def next_permutation(permutation):
    for i in range(len(permutation) - 2, -1, -1):
        if permutation[i] < permutation[i + 1]:
            k = i
            break
    else:
        return

    for i in range(len(permutation) - 1, k, -1):
        if permutation[k] < permutation[i]:
            r = i
            break

    permutation = list(permutation)
    permutation[k], permutation[r] = permutation[r], permutation[k]
    return ''.join(permutation[0:k + 1] + list(reversed(permutation[k + 1:])))


class ImplicitTreap:
    class ImplicitTreapNode:
        def __init__(self, y, value, left=None, right=None):
            self.y = y
            self.value = value
            self.left = left
            self.right = right
            self.size = 1

        def recalc(self):
            self.size = (self.left.size if self.left is not None else 0) + \
                        (self.right.size if self.right is not None else 0) + 1

    def __init__(self, values=None):
        self.root = None
        for value in values:
            self.append(value)

    @staticmethod
    def _merge(left, right):
        if left is None:
            return right
        if right is None:
            return left

        if left.y > right.y:
            left.right = ImplicitTreap._merge(left.right, right)
            result = left
        else:
            right.left = ImplicitTreap._merge(left, right.left)
            result = right

        result.recalc()
        return result

    @staticmethod
    def _split(node, index):
        if node is None:
            return None, None

        left_size = node.left.size if node.left is not None else 0
        if index >= left_size + 1:
            left, right = ImplicitTreap._split(node.right, index - left_size - 1)
            node.right = left
            node.recalc()
            return node, right
        else:
            left, right = ImplicitTreap._split(node.left, index)
            node.left = right
            node.recalc()
            return left, node

    def __getitem__(self, index):
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

    def insert(self, index, value):
        if self.root is None:
            self.root = ImplicitTreap.ImplicitTreapNode(random.random(), value)
            return

        if index > self.root.size:
            raise Exception('Index out of range')

        left, right = ImplicitTreap._split(self.root, index)
        node = ImplicitTreap.ImplicitTreapNode(random.random(), value)
        self.root = ImplicitTreap._merge(ImplicitTreap._merge(left, node), right)

    def append(self, value):
        self.insert(self.root.size if self.root is not None else 0, value)

    def pop(self, index=None):
        if index is None:
            index = self.root.size - 1

        left, right = ImplicitTreap._split(self.root, index)
        _, right = ImplicitTreap._split(right, 1)
        self.root = ImplicitTreap._merge(left, right)

    def __len__(self):
        return self.root.size if self.root is not None else 0

    def __str__(self):
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



