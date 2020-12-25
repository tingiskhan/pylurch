import pandas as pd
from typing import Dict, Union, Tuple, Any, Generic, TypeVar
import numpy as np
import dill
import onnxruntime as rt
import git
from pylurch.contract import enums
from .container import InferenceContainer, TModel
from .types import FrameOrArray


TOutput = TypeVar("TOutput")
T = InferenceContainer[TModel]
U = InferenceContainer[TOutput]


class InferenceModelBlueprint(Generic[TModel, TOutput]):
    def __init__(self, name: str = None, base=None):
        """
        Defines a base class for performing inference etc.
        """

        self._name = name or self.__class__.__name__
        self._base = base  # type: InferenceModelBlueprint

    @property
    def base(self):
        return self._base

    @property
    def is_derived(self) -> bool:
        return self._base is not None

    @staticmethod
    def git_commit_hash():
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def name(self):
        return self._name

    def serializer_backend(self) -> enums.SerializerBackend:
        raise NotImplementedError()

    def parse_x_y(self, x: str, y: str = None, **kwargs) -> Tuple[FrameOrArray, ...]:
        parsed_x = pd.read_json(x, **kwargs).sort_index()
        parsed_y = y if y is None else pd.read_json(y, **kwargs).sort_index()

        return parsed_x, parsed_y

    def make_model(self, **kwargs) -> T:
        raise NotImplementedError()

    def serialize(self, container: T, *args: Tuple[Any, ...], x: FrameOrArray = None, y: FrameOrArray = None) -> bytes:
        raise NotImplementedError()

    def deserialize(self, byte_string: bytes) -> U:
        if self.serializer_backend() == enums.SerializerBackend.Custom:
            raise NotImplementedError("Please override this method!")

        if self.serializer_backend() == enums.SerializerBackend.ONNX:
            return InferenceContainer(rt.InferenceSession(byte_string))

        if self.serializer_backend() == enums.SerializerBackend.Dill:
            return InferenceContainer(dill.loads(byte_string))

    def fit(self, container: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]):
        raise NotImplementedError()

    def update(self, container: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]):
        raise ValueError("This model does not support updating!")

    def predict(self, container: U, x: FrameOrArray, **kwargs: Dict[str, object]) -> FrameOrArray:
        if self.serializer_backend() != enums.SerializerBackend.ONNX:
            raise NotImplementedError(f"You must override the method yourself!")

        inp_name = container.model.get_inputs()[0].name
        label_name = container.model.get_outputs()[0].name

        res = container.model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(res, index=x.index, columns=["y"])

        return res
