from .enums import SerializerBackend, ModelStatus
import pandas as pd
from typing import Dict
from logging import Logger
import numpy as np
from .model_managers import BaseModelManager
import dill
import onnxruntime as rt


class InferenceModel(object):
    def __init__(self, name: str, logger: Logger, model_manager: BaseModelManager):
        """
        Defines a base class for performing inference etc.
        """

        self._name = name
        self.logger = logger
        self._model_manager = model_manager

    def set_model_manager(self, model_manager: BaseModelManager):
        if self._model_manager is None:
            self._model_manager = model_manager

    @property
    def model_manager(self):
        return self._model_manager

    def name(self):
        return self._name

    def serializer_backend(self) -> SerializerBackend:
        raise NotImplementedError()

    def add_metadata(self, model: object, **kwargs: Dict[str, str]) -> Dict[str, str]:
        """
        Allows user to add string meta data associated with the model. Is called when model is done.
        :param model: The instantiated model.
        :param kwargs: The key worded arguments associated with the model
        """

        return dict()

    def _run_model(self, func: callable, model: object, x: pd.DataFrame, key: str, **kwargs):
        """
        Utility function for running the model and handling persisting/exceptions.
        :param func: The function to apply
        :param model: The model
        :param x: The data
        :param key: The key
        """

        self.logger.info(f"Starting training of '{self.name()}' with '{key}' and using {x.shape[0]} observations")

        # ===== Fit/update ===== #
        try:
            res = func(model, x, **kwargs)
            self.logger.info(f"Successfully finished training '{self.name()}' with '{key}'")
        except Exception as e:
            self.logger.exception(f"Failed '{self.name()}' with '{key}'", e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())
            return False

        # ===== Save ===== #
        try:
            self.logger.info(f"Now trying to serialize '{self.name()}' with '{key}'")
            bytestring = self.serialize(res, x)

            meta_data = self.add_metadata(res, x=x, **kwargs)

            self.logger.info(f"Now trying to persist '{self.name()}' with '{key}'")
            self.model_manager.save(self.name(), key, bytestring, self.serializer_backend(), meta_data=meta_data)
            self.logger.info(f"Successfully persisted '{self.name()}' with '{key}'")

        except Exception as e:
            self.logger.exception(f'Failed persisting {key}', e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())

            return False

        return True

    def do_run(self, model: object, x: pd.DataFrame, key: str, **kwargs) -> bool:
        return self._run_model(self.fit, model, x, key, **kwargs)

    def do_update(self, model: object, x: pd.DataFrame, key: str, **kwargs) -> bool:
        return self._run_model(self.update, model, x, key, **kwargs)

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

        if self.serializer_backend() == SerializerBackend.Custom:
            raise NotImplementedError('Please override this method!')
        if self.serializer_backend() == SerializerBackend.ONNX:
            return rt.InferenceSession(bytestring)
        elif self.serializer_backend() == SerializerBackend.Dill:
            return dill.loads(bytestring)

    def fit(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None, **kwargs: Dict[str, object]) -> object:
        """
        Fits the model
        :param model: The model to fit
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise NotImplementedError()

    def update(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None, **kwargs: Dict[str, object]) -> object:
        """
        Fits the model
        :param model: The model to update
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise ValueError('This model does not support updating!')

    def predict(self, model: object, x: pd.DataFrame, **kw: Dict[str, object]) -> pd.DataFrame:
        """
        Return the prediction.
        :param model: The model to use for predicting
        :param x: The data to predict for
        """

        if self.serializer_backend() != SerializerBackend.ONNX:
            raise NotImplementedError(f'You must override the method yourself!')

        inp_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

        res = model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        return pd.DataFrame(res, index=x.index, columns=['y'])

    def check_status(self, key: str) -> ModelStatus:
        """
        Helper function for checking the status of the model.
        :param key: The key of the model
        """

        return self.model_manager.check_status(self.name(), key, self.serializer_backend())

    def pre_model_start(self, key: str):
        return self.model_manager.pre_model_start(self.name(), key, self.serializer_backend())

    def load(self, key: str) -> object:
        """
        Method for loading the model.
        :param key: The key
        """

        obj = self.model_manager.load(self.name(), key, self.serializer_backend())

        if obj is None:
            return None

        return self.deserialize(obj)

    def delete(self, key: str):
        """
        Method for deleting a model.
        :param key: The key
        """

        self.model_manager.delete(self.name(), key, self.serializer_backend())
        return self