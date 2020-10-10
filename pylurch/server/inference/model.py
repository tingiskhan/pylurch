import pandas as pd
from typing import Dict, Union
import numpy as np
import dill
import onnxruntime as rt
from pylurch.contract import enums


FrameOrArray = Union[pd.DataFrame, np.ndarray]


class InferenceModel(object):
    def __init__(self, name: str = None, base=None):
        """
        Defines a base class for performing inference etc.
        """

        self._name = name or self.__class__.__name__
        self._base = base  # type: InferenceModel

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

    def parse_data(self, data: str, **kwargs) -> FrameOrArray:
        """
        Method for parsing data.
        :param data: The data in string format
        """

        return pd.read_json(data, **kwargs).sort_index()

    def make_model(self, **kwargs) -> object:
        """
        Creates the model
        :param kwargs: Any key worded arguments passed on instantiation to the model.
        """

        raise NotImplementedError()

    def serialize(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None) -> bytes:
        """
        Serialize model to byte string.
        :param model: The model to convert
        :param x: The data used for training
        :param y: The data used for training
        """

        raise NotImplementedError()

    def deserialize(self, bytestring: bytes) -> object:
        """
        Method for deserializing the model. Can be overridden if custom serializer.
        :param bytestring: The byte string
        """

        if self.serializer_backend() == enums.SerializerBackend.Custom:
            raise NotImplementedError('Please override this method!')
        if self.serializer_backend() == enums.SerializerBackend.ONNX:
            return rt.InferenceSession(bytestring)
        elif self.serializer_backend() == enums.SerializerBackend.Dill:
            return dill.loads(bytestring)

    def fit(self, model: object, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> object:
        """
        Fits the model
        :param model: The model to fit
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise NotImplementedError()

    def update(self, model: object, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> object:
        """
        Updates the model
        :param model: The model to update
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise ValueError('This model does not support updating!')

    def predict(self, model: object, x: FrameOrArray, **kw: Dict[str, object]) -> FrameOrArray:
        """
        Return the prediction.
        :param model: The model to use for predicting
        :param x: The data to predict for
        """

        if self.serializer_backend() != enums.SerializerBackend.ONNX:
            raise NotImplementedError(f'You must override the method yourself!')

        inp_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

        res = model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(res, index=x.index, columns=['y'])

        return res
