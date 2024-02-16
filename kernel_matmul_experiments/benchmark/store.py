from typing import Any, Self


class HierarchicalStore:
    """
    Special dict-like class that allows to store values in a hierarchical way.

    You can index into the store using the `[]` operator.
    The behavior depends on the current state:

    - If the current state is a leaf, it is returned. Thus, once a value is set at one level of
      the tree, all children have this value, too.
    - Otherwise, an implicit node is created and returned. This node is not stored in the tree
      until it is assigned a value.

    You can assign a value using the `value` property. This will create a leaf at the current
    position in the tree and store the value there. If the current state is an implicit node,
    it will be converted to a normal node and all parents will be created.

    You can convert the store to a dict using the `to_dict()` method. This will raise an error
    if the current state is an implicit node. You can also create a store from a dict using the
    `from_dict()` class method.
    """

    def __init__(self):
        self._state = _Node()

    @property
    def exists(self) -> bool:
        return not isinstance(self._state, _Implicit)

    @property
    def has_value(self) -> bool:
        return isinstance(self._state, _Leaf)

    @property
    def value(self) -> Any:
        if not self.has_value:
            raise KeyError("No value stored at this key.")
        return self._state.value

    @value.setter
    def value(self, value: Any):
        if isinstance(value, dict):
            raise ValueError("Cannot set a dict as value.")
        if isinstance(self._state, _Implicit):
            self._state.parent._register_child(self._state.key, self)
        self._state = _Leaf(value)

    def __getitem__(self: Self, key: Any) -> Self:
        if isinstance(self._state, _Leaf):
            return self
        if isinstance(self._state, _Node) and key in self._state.children:
            return self._state.children[key]
        child = type(self)()
        child._state = _Implicit(self, key)
        return child

    def _register_child(self, key: Any, child: Self) -> None:
        if isinstance(self._state, _Leaf):
            raise KeyError("This node is a leaf.")
        if isinstance(self._state, _Implicit):
            self._state.parent._register_child(self._state.key, self)
            self._state = _Node()
        if key in self._state.children:
            raise KeyError("This key already exists.")
        self._state.children[key] = child

    def to_dict(self) -> Any:
        if isinstance(self._state, _Implicit):
            raise KeyError("This node is implicit.")
        if isinstance(self._state, _Leaf):
            return self._state.value
        return {key: child.to_dict() for key, child in self._state.children.items()}

    @classmethod
    def from_dict(cls: type[Self], data: Any) -> Self:
        self = cls()
        if isinstance(data, dict):
            for key, value in data.items():
                self._state.children[key] = cls.from_dict(value)
        else:
            self._state = _Leaf(data)
        return self


class _State:
    ...


class _Node(_State):
    def __init__(self: Self):
        self.children: dict[Any, Self] = {}


class _Leaf(_State):
    def __init__(self, value: Any):
        self.value = value


class _Implicit(_State):
    def __init__(self, parent: HierarchicalStore, key: Any):
        self.parent = parent
        self.key = key
