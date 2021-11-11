from abc import ABCMeta


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
    ):
        NotImplemented
