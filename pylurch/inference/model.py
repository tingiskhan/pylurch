import pandas as pd
from typing import Dict, Union, TypeVar, Tuple, Any
import numpy as np
import dill
import onnxruntime as rt
from pylurch.contract import enums


FrameOrArray = Union[pd.DataFrame, np.ndarray]
T = TypeVar("T")


class InferenceModel(object):
    def __init__(self, name: str = None, base=None):
        """
        Defines a base class for performing inference etc.
        """

        self._name = name or self.__class__.__name__
        self._base = base  # type: InferenceModel
        self.model = None

    @property
    def base(self):
        return self._base

    @property
    def is_derived(self) -> bool:
        return self._base is not None

    def name(self):
        return self._name

    def serializer_backend(self) -> enums.SerializerBackend:
        raise NotImplementedError()

    def add_metadata(self, model: object, **kwargs: Dict[str, str]) -> Dict[str, str]:
        """
        Allows user to add string meta data associated with the model. Is called when model is done.
        :param model: The instantiated model.
        :param kwargs: The key worded arguments associated with the model
        """

        return dict()

    def parse_x_y(self, x: str, y: str = None, **kwargs) -> Tuple[FrameOrArray, ...]:
        parsed_x = pd.read_json(x, **kwargs).sort_index()
        parsed_y = y if y is None else pd.read_json(y, **kwargs).sort_index()

        return parsed_x, parsed_y

    def make_model(self, **kwargs) -> T:
        raise NotImplementedError()

    def serialize(self, model: T, x: pd.DataFrame, y: pd.DataFrame = None) -> bytes:
        raise NotImplementedError()

    def deserialize(self, bytestring: bytes) -> T:
        if self.serializer_backend() == enums.SerializerBackend.Custom:
            raise NotImplementedError("Please override this method!")
        if self.serializer_backend() == enums.SerializerBackend.ONNX:
            return rt.InferenceSession(bytestring)
        elif self.serializer_backend() == enums.SerializerBackend.Dill:
            return dill.loads(bytestring)

    def fit(self, model: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> T:
        raise NotImplementedError()

    def update(self, model: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> T:
        raise ValueError("This model does not support updating!")

    def predict(self, model: T, x: FrameOrArray, **kw: Dict[str, object]) -> FrameOrArray:
        if self.serializer_backend() != enums.SerializerBackend.ONNX:
            raise NotImplementedError(f"You must override the method yourself!")

        inp_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

        res = model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(res, index=x.index, columns=["y"])

        return res

    def do_make_model(self, **model_kwargs):
        self.model = self.make_model(**model_kwargs)

    def do_fit(
        self,
        x: FrameOrArray,
        y: FrameOrArray = None,
        model_kwargs: Dict[str, Any] = None,
        alg_kwargs: Dict[str, Any] = None,
    ):
        self.do_make_model(**(model_kwargs or dict()))
        return self.fit(self.model, x, y, **(alg_kwargs or dict()))
