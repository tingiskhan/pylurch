import pandas as pd
from .utils import hash_series, custom_error
from hashlib import sha256
from .enums import ModelStatus, SerializerBackend
import numpy as np
from .metaresource import BaseModelResource
import dill
import onnxruntime as rt
from typing import Dict
from flask_executor.futures import Future


class ModelResource(BaseModelResource):
    def make_model(self, **kwargs: Dict[str, object]) -> BaseModelResource:
        """
        Creates the model to be used.
        :param kwargs: Any key-worded arguments passed to the model.
        """

        raise NotImplementedError()

    def serializer_backend(self) -> SerializerBackend:
        """
        Returns the backend used by the backend.
        """

        raise NotImplementedError()

    def done_callback(self, fut: Future, key: str, x: pd.DataFrame, **kwargs: Dict[str, object]):
        """
        Callback for when futures is done with work.
        :param fut: The future
        :param key: The key associated with the model
        :param x: The data
        """

        res = fut.result()

        if res is None:
            return

        self.logger.info(f'Successfully trained {key}, now trying to persist')

        try:
            bytestring = self.serialize(res, x)

            meta_data = self.add_metadata(res, x=x, **kwargs)
            self.save_model(key, bytestring, meta_data=meta_data)

            self.logger.info(f'Successfully persisted {key}')
        except Exception as e:
            self.logger.exception(f'Failed persisting {key}', e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())
        finally:
            self.executor.futures.pop(self._make_executor_key(key))

    def serialize(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None) -> bytes:
        """
        Serialize model to byte string.
        :param model: The model to convert
        :param x: The data used for training
        :param y: The data used for training
        """

        raise NotImplementedError()

    def save_model(self, key: str, mod: bytes, meta_data: Dict[str, str] = None) -> None:
        """
        Method for saving the model.
        :param key: The key
        :param mod: The model
        :param meta_data: Whether to add any metadata associated with the model
        """

        self.model_manager.save(self.name(), key, mod, self.serializer_backend(), meta_data=meta_data)

    def load_model(self, key: str) -> object:
        """
        Method for loading the model.
        :param key: The key
         The model object
        """

        obj = self.model_manager.load(self.name(), key, self.serializer_backend())

        if obj is None:
            return None

        return self.deserialize(obj)

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
        :param model: The model to use
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise NotImplementedError()

    def update(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None, **kwargs: Dict[str, object]) -> object:
        """
        Fits the model
        :param model: The model to use
        :param x: The data
        :param y: The response data (if any)
        :param kwargs: Any additional key worded arguments
        """

        raise ValueError('This model does not support updating!')

    def predict(self, mod: object, x: pd.DataFrame, **kw: Dict[str, object]) -> pd.DataFrame:
        """
        Return the prediction.
        :param mod: The model
        :param x: The data to predict for
        """

        if self.serializer_backend() != SerializerBackend.ONNX:
            raise NotImplementedError(f'You must override the method yourself!')

        inp_name = mod.get_inputs()[0].name
        label_name = mod.get_outputs()[0].name

        res = mod.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        return pd.DataFrame(res, index=x.index, columns=['y'])

    def run_model(self, func: callable, model: object, x: pd.DataFrame, key: str, **kwargs):
        """
        Utility function for running the model and handling persisting/exceptions.
        :param func: The function to apply
        :param model: The model
        :param x: The data
        :param key: The key
        """

        self.logger.info(f'Starting training of {self.name()} using {x.shape[0]} observations')

        try:
            return func(model, x, **kwargs)
        except Exception as e:
            self.logger.exception(f'Failed task with key: {key}', e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())
            return None

    def parse_data(self, data: str, **kwargs) -> pd.DataFrame:
        """
        Method for parsing data.
        :param data: The data in string format
         Data in required format
        """

        return pd.read_json(data, **kwargs).sort_index()

    def _make_executor_key(self, key: str) -> str:
        return f'{self.name()}-{key}'

    def check_model_status(self, key: str) -> ModelStatus:
        """
        Helper function for checking the status of the model.
        :param key: The key of the model
        """

        return self.model_manager.check_status(self.name(), key, self.serializer_backend())

    def name(self) -> str:
        """
        The name of the model to use.
        """

        raise NotImplementedError()

    def add_metadata(self, model: object, **kwargs: Dict[str, str]) -> Dict[str, str]:
        """
        Allows user to add string meta data associated with the model. Is called when model is done.
        :param model: The instantiated model.
        :param kwargs: The key worded arguments associated with the model
        """

        return dict()

    @custom_error
    def _put(self, x: str, orient: str, name: str, y: str = None, modkwargs: Dict[str, object] = None,
             algkwargs: Dict[str, object] = None, retrain: bool = False):
        # ===== Get data ===== #
        x = self.parse_data(x, orient=orient)

        # ===== Generate model key ===== #
        dkey = sha256(name.lower().encode()).hexdigest()

        # ===== Check status ===== #
        status = self.check_model_status(dkey)

        if status not in (ModelStatus.Unknown, ModelStatus.Failed) and not retrain:
            self.logger.info('Model already exists, and no retrain requested')
            return {'status': status, 'model_key': dkey}

        # ===== Define model ===== #
        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()

        if y is not None:
            akws['y'] = self.parse_data(y, orient=orient)

        model = self.make_model(**modkwargs)

        if status == ModelStatus.Running:
            if retrain:
                return {'status': status, 'model_key': dkey}, 400

            return {'status': status, 'model_key': dkey}

        key = self._make_executor_key(dkey)

        # ===== Let it persist run first ===== #
        self.model_manager.pre_model_start(self.name(), dkey, self.serializer_backend())

        # ===== Start background task ===== #
        futures = self.executor.submit_stored(key, self.run_model, self.fit, model, x, dkey, **akws)
        futures.add_done_callback(lambda u: self.done_callback(u, dkey, x=x, **akws))

        return {'status': self.check_model_status(dkey), 'model_key': dkey}

    @custom_error
    def _post(self, model_key: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, object]):
        status = self.check_model_status(model_key)

        if status != ModelStatus.Done:
            return {'data': None, 'orient': orient}, 400

        mod = self.load_model(model_key)

        if mod is None:
            return {'data': None, 'orient': orient}, 400

        self.logger.info(f'Predicting values using model {self.name()}')

        x_hat = self.predict(mod, self.parse_data(x, orient=orient), **kwargs)

        if as_array:
            x_resp = x_hat.values.tolist()
        else:
            x_resp = x_hat.to_json(orient=orient)

        resp = {
            'data': x_resp,
            'orient': orient,
        }

        return resp

    @custom_error
    def _get(self, model_key):
        status = self.check_model_status(model_key)

        return {'status': status}

    @custom_error
    def _patch(self, model_key: str, x: str, orient: str, y: str = None):
        status = self.check_model_status(model_key)

        if status != ModelStatus.Done:
            return {'status': status}

        model = self.load_model(model_key)
        x = self.parse_data(x, orient=orient)

        kwargs = dict()
        if y is not None:
            kwargs['y'] = self.parse_data(y, orient=orient)

        key = self._make_executor_key(model_key)

        # ===== Let it persist run first ===== #
        self.model_manager.pre_model_start(self.name(), key, self.serializer_backend())

        # ===== Start background task ===== #
        futures = self.executor.submit_stored(key, self.run_model, self.update, model, x, model_key, **kwargs)
        futures.add_done_callback(lambda u: self.done_callback(u, model_key, x=x))

        self.logger.info(f'Started updating of model {self.name()} using {x.shape[0]} new observations')

        return {'status': self.check_model_status(model_key)}

    @custom_error
    def _delete(self, model_key):
        self.logger.info(f'Deleting model with key: {model_key}')

        self.model_manager.delete(self.name(), model_key, self.serializer_backend())

        self.logger.info(f'Successfully deleted model with key: {model_key}')

        return {'status': ModelStatus.Done}