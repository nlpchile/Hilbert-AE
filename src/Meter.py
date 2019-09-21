class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, val=1):
        raise NotImplementedError

    @property
    def avg(self):
        return 0


# TODO : It could work with dictionaries,
# or we could extend it to a "Multi" Accumulator
class Accumulator(Meter):
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.accumulated: float = 0.0
        self.n: int = 0

    def __call__(self, value: float):
        return self.__class__.update(self, value=value)

    def update(self, value: float = 0.0) -> None:
        self.accumulated += value
        self.n += 1

    @property
    def avg(self) -> float:
        return self.accumulated / (1.0 * self.n)
