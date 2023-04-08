class State(int):
    is_end = False

    def __new__(cls, val=None, is_end: bool = False, *args, **kwargs):
        object = super().__new__(cls, val, *args, **kwargs)
        object.is_end = is_end
        return object


class GameState(State):
    pass


class InfoState(State):
    pass


