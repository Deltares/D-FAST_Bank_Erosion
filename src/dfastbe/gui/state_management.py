from __future__ import annotations


class StateStore(dict):
    """Singleton dictionary for shared GUI state.

    The GUI initializes the singleton exactly once via `StateStore.initialize()`
    (called from `GUI.__init__`). All other modules should call
    `StateStore.instance()` to access the same shared dictionary. If the GUI has
    not been constructed yet, `instance()` raises a RuntimeError to make the
    initialization order explicit.
    """
    _instance: "StateStore | None" = None

    @classmethod
    def initialize(cls) -> "StateStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def instance(cls) -> "StateStore":
        if cls._instance is None:
            raise RuntimeError("StateStore not initialized. Create GUI() first.")
        return cls._instance


StateManagement: StateStore | None = None
