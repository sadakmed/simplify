from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
        self,
    ):
        NotImplemented
