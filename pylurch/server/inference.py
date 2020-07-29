import pandas as pd
from typing import Dict, Union
from logging import Logger
import numpy as np
from pylurch.contract.interfaces import DatabaseInterface
import dill
import onnxruntime as rt
from pylurch.contract import database as db, enums
from ..utils import make_base_logger


FrameOrArray = Union[pd.DataFrame, np.ndarray]


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
        return self._intf.make_interface(db.TrainingSession)

    @property
    def mod_intf(self):
        return self._intf.make_interface(db.Model)

    @property
    def md_intf(self):
        return self._intf.make_interface(db.MetaData)

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

    def _run_model(self, func: callable, model: object, x: FrameOrArray, name: str, **kwargs) -> db.TrainingSession:
        """
        Utility function for running the model and handling persisting/exceptions.
        :param func: The function to apply
        :param model: The model
        :param x: The data
        :param key: The key
        """

        if self._base is not None:
            raise NotImplementedError(f"'{self.name()}' inherits from '{self._base.name()}' and is thus not trainable!")

        db_model = self._get_model()
        self.logger.info(f"Starting training of '{self.name()}' with '{name}' and using {x.shape[0]} observations")

        # ===== Fit/update ===== #
        res = func(model, x, **kwargs)
        self.logger.info(f"Successfully finished training '{self.name()}' with '{name}'")

        # ===== Save model ===== #
        self.logger.info(f"Now trying to serialize '{self.name()}' with '{name}'")
        bytestring = self.serialize(res, x)

        self.logger.info(f"Now trying to persist '{self.name()}' with '{name}'")
        session = db.TrainingSession(
            model_id=db_model.id,
            session_name=name,
            backend=self.serializer_backend(),
            byte_string=bytestring
        )

        session = self.ts_intf.create(session)
        self.logger.info(f"Successfully persisted '{self.name()}' with '{name}'")

        # ==== Save meta data ===== #
        meta_data = self.add_metadata(res, x=x, **kwargs)
        md = [db.MetaData(session_id=session.id, key=k, value=v) for k, v in meta_data.items()]

        if any(md):
            self.md_intf.create(md)

        return session

    def do_run(self, model: object, x: FrameOrArray, name: str, **kwargs) -> db.TrainingSession:
        return self._run_model(self.fit, model, x, name=name, **kwargs)

    def do_update(self, model: object, x: FrameOrArray, old_name: str, name: str, **kwargs) -> db.TrainingSession:
        res = self._run_model(self.update, model, x, name=name, **kwargs)

        dbmod = self.mod_intf.get(lambda u: u.name == self.name(), one=True)
        old = self.ts_intf.get(lambda u: (u.session_name == old_name) & (u.model_id == dbmod.id), one=True)

        # ===== Create link ===== #
        link = db.UpdatedSession(
            base=old.id,
            new=res.id
        )

        result = self._intf.make_interface(db.UpdatedSession).create(link)

        return res

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

    def fit(self, model: object, x: FrameOrArray, y: FrameOrArray, **kwargs: Dict[str, object]) -> object:
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

    def _get_model(self) -> db.Model:
        """
        Initialize the training of the model.
        :param key: The key
        """
        model = self.mod_intf.get(lambda u: u.name == self.name(), one=True)

        if model is None:
            model = self.mod_intf.create(db.Model(name=self.name()))

        return model

    def load(self, name: str) -> object:
        """
        Method for loading the latest successfully trained model with key.
        :param name: The name of the session
        """

        if self._base is not None:
            self.logger.info(f"'{self.name()}' is derived and loads model from {self._base.name()}")
            return self._base.load(name)

        mod = self.mod_intf.get(lambda u: u.name == self.name(), one=True)
        if mod is None:
            return None

        session = self.ts_intf.get(lambda u: u.session_name == name, one=True)
        if session is None:
            return session

        return self.deserialize(session.byte_string)
