from abc import ABCMeta, abstractmethod



class _BaseData(metaclass=ABCMeta):


    @abstractmethod
    def encode(self,sentence):
        ...

    @abstractmethod
    def decode(self, output):
        ...

    def encode_batch(self, sentences):
        return [self.encode(sentence) for sentence in sentences]

    def decode_batch(self, outputs):
        return [self.decode(output) for output in outputs]


class _BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def simplify(self,*args, **kwargs):
        ...

    def __call__(self,*args, **kwargs):
        return self.simplify(*args, **kwargs)

    def forward(self,*args, **kwargs):
        return self.simplify(*args, **kwargs)


