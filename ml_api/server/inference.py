import pandas as pd
from typing import Dict
from logging import Logger
import numpy as np
from ml_api.contract.interfaces import DatabaseInterface
import dill
import onnxruntime as rt
from ml_api.contract import schemas as sc, database as db, enums
from datetime import datetime
from ..utils import make_base_logger


class InferenceModel(object):
    def __init__(self, intf: DatabaseInterface, name: str = None, logger: Logger = None, base=None):
        """
        Defines a base class for performing inference etc.
        """

        self._name = name or self.__class__.__name__
        self._base = base  # type: InferenceModel
        self.logger = logger or make_base_logger(self.name())
        self._intf = intf

    @property
    def interface(self):
        return self._intf

    def set_intf(self, intf: DatabaseInterface):
        self._intf = intf

    @property
    def ts_intf(self):
        return self._intf.make_interface(sc.TrainingSessionSchema)

    @property
    def mod_intf(self):
        return self._intf.make_interface(sc.ModelSchema)

    @property
    def md_intf(self):
        return self._intf.make_interface(sc.MetaDataSchema)

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

    def _run_model(self, func: callable, model: object, x: pd.DataFrame, session: db.TrainingSession, **kwargs):
        """
        Utility function for running the model and handling persisting/exceptions.
        :param func: The function to apply
        :param model: The model
        :param x: The data
        :param key: The key
        """

        if self._base is not None:
            raise NotImplementedError(f"'{self.name()}' inherits from '{self._base.name()}' and is thus not trainable!")

        key = session.hash_key
        self.logger.info(f"Starting training of '{self.name()}' with '{key}' and using {x.shape[0]} observations")

        # ===== Fit/update ===== #
        try:
            res = func(model, x, key=key, **kwargs)
            self.logger.info(f"Successfully finished training '{self.name()}' with '{key}'")
        except Exception as exc:
            self.logger.exception(f"Failed '{self.name()}' with '{key}'", exc)

            session.status = enums.ModelStatus.Failed
            session.end_time = datetime.now()
            self.ts_intf.update(session)

            return False

        # ===== Save model ===== #
        try:
            self.logger.info(f"Now trying to serialize '{self.name()}' with '{key}'")
            bytestring = self.serialize(res, x)

            self.logger.info(f"Now trying to persist '{self.name()}' with '{key}'")

            session.byte_string = bytestring
            session.status = enums.ModelStatus.Done
            session.end_time = datetime.now()

            session = self.ts_intf.update(session)[0]

            self.logger.info(f"Successfully persisted '{self.name()}' with '{key}'")

        except Exception as exc:
            self.logger.exception(f"Failed persisting '{key}'", exc)

            session.byte_string = None
            session.status = enums.ModelStatus.Failed
            session.end_time = datetime.now()

            self.ts_intf.update(session)

            return False

        # ==== Save meta data ===== #
        try:
            meta_data = self.add_metadata(res, x=x, **kwargs)
            md = [db.MetaData(session_id=session.id, key=k, value=v) for k, v in meta_data.items()]
            if any(md):
                self.md_intf.create(md)

        except Exception as exc:
            self.logger.exception(f"Failed persisting meta data for '{key}', but setting", exc)

            return False

        return True

    def do_run(self, model: object, x: pd.DataFrame, session: db.TrainingSession, **kwargs) -> bool:
        return self._run_model(self.fit, model, x, session, **kwargs)

    def do_update(self, model: object, x: pd.DataFrame, session: db.TrainingSession, **kwargs) -> bool:
        return self._run_model(self.update, model, x, session, **kwargs)

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

    def fit(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None, key: str = None,
            **kwargs: Dict[str, object]) -> object:
        """
        Fits the model
        :param model: The model to fit
        :param x: The data
        :param y: The response data (if any)
        :param key: The key of the current model instance
        :param kwargs: Any additional key worded arguments
        """

        raise NotImplementedError()

    def update(self, model: object, x: pd.DataFrame, y: pd.DataFrame = None, key: str = None,
               **kwargs: Dict[str, object]) -> object:
        """
        Updates the model
        :param model: The model to update
        :param x: The data
        :param y: The response data (if any)
        :param key: The key of the current model instance
        :param kwargs: Any additional key worded arguments
        """

        raise ValueError('This model does not support updating!')

    def predict(self, model: object, x: pd.DataFrame, **kw: Dict[str, object]) -> pd.DataFrame:
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

        return pd.DataFrame(res, index=x.index, columns=['y'])

    def check_status(self, key: str) -> enums.ModelStatus:
        """
        Helper function for checking the status of the latest model with key.
        :param key: The key of the model
        """

        if self._base is not None:
            return self._base.check_status(key)

        model = self.mod_intf.get(lambda u: u.name == self.name(), one=True)

        if model is None:
            return enums.ModelStatus.Unknown

        session = self.ts_intf.get(lambda u: (u.model_id == model.id) & (u.hash_key == key))
        latest = sorted(session, key=lambda u: u.id, reverse=True)

        return latest[0].status if any(latest) else enums.ModelStatus.Unknown

    def initialize_training(self, key: str) -> db.TrainingSession:
        """
        Initialize the training of the model.
        :param key: The key
        """
        model = self.mod_intf.get(lambda u: u.name == self.name(), one=True)

        if model is None:
            model = self.mod_intf.create(db.Model(name=self.name()))

        session = db.TrainingSession(
            model_id=model.id,
            hash_key=key,
            start_time=datetime.now(),
            end_time=datetime.max,
            status=enums.ModelStatus.Running,
            backend=self.serializer_backend()
        )

        return self.ts_intf.create(session)

    def load(self, key: str) -> object:
        """
        Method for loading the latest successfully trained model with key.
        :param key: The key
        """

        if self._base is not None:
            self.logger.info(f"'{self.name()}' is derived and loads model from {self._base.name()}")
            return self._base.load(key)

        mod = self.mod_intf.get(lambda u: u.name == self.name(), one=True)
        if mod is None:
            return None

        sessions = self.ts_intf.get(
            lambda u: (u.model_id == mod.id) & (u.hash_key == key) & (u.status == enums.ModelStatus.Done)
        )

        latest = sorted(sessions, key=lambda u: u.id, reverse=True)

        if not any(latest):
            return None

        return self.deserialize(latest[0].byte_string)
