# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from collections import deque


# TODO: Benchmark which of these is faster. If `FlatScopedTable` is faster, 
# evaluate if worths using it.
#
# The `ScopedTable` have a better balance between efficiency and simplicity.
# Its main issue is O(n) lookup, since in the worst case we might have to look
# for a symbol in `n` scopes. (Note: typically, due to locality of references in
# real code scenarios, this might not play a decisive role.)
#
# As the name suggests `FlatScopedTable` store symbols in a single flat 
# dictionary, and thus lookup is O(1). The trade-offs here are:
#   * More complex key managment
#   * If the symbol tables grows too large, it could affect performance 
#     (How large to see an effect? I don't know :)
#   * It is overall less intuitive than `ScopedTable`

class Secope:
    def __init__(self):
        self.bindings = {}

class ScopedTable:
    def __init__(self):
        self.scopes = deque([{}])  # Use deque for efficient push/pop
        self.current_scope = self.scopes[-1]  # Direct reference to current scope

    def push_scope(self):
        new_scope = {}
        self.scopes.append(new_scope)
        self.current_scope = new_scope

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
            self.current_scope = self.scopes[-1]

    def depth(self):
        return len(self.scopes)

    def insert(self, name, value):
        self.current_scope[name] = value

    def lookup(self, name):
        for scope in reversed(self.scopes):
            value = scope.get(name)
            if value is not None:
                return value
        return None

    def lookup_or_insert(self, name, default_value):
        for scope in reversed(self.scopes):
            value = scope.get(name)
            if value is not None:
                return value
        self.current_scope[name] = default_value
        return self.current_scope[name]

    def update(self, name, value):
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return True
        return False


class FlatScopedTable:
    def __init__(self):
        self.symbols = {}
        self.scopes = []

    def push_scope(self):
        self.scopes.append(len(self.symbols))

    def pop_scope(self):
        if self.scopes:
            scope_start = self.scopes.pop()
            self.symbols = {k: v for k, v in self.symbols.items() if v[1] < scope_start}

    def insert(self, name, value):
        scope_depth = len(self.scopes)
        self.symbols[f"{name}_{scope_depth}"] = (value, len(self.symbols))

    def lookup(self, name):
        for depth in range(len(self.scopes), -1, -1):
            key = f"{name}_{depth}"
            value = self.symbols.get(key)
            if value is not None:
                return value[0]
        return None

    def update(self, name, value):
        for depth in range(len(self.scopes), -1, -1):
            key = f"{name}_{depth}"
            if key in self.symbols:
                self.symbols[key] = (value, self.symbols[key][1])
                return True
        return False
