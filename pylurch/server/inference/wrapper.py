import pandas as pd
from typing import Union, Callable, Any, Dict, Optional, List
from logging import Logger
import numpy as np
from pylurch.contract.interface import SessionInterface
from pylurch.contract import database as db
from pyalfred.server.utils import make_base_logger
from .model import InferenceModel, T


FrameOrArray = Union[pd.DataFrame, np.ndarray]
Func = Callable[[T, FrameOrArray, Optional[FrameOrArray], Dict[str, Any]], Any]


class ModelWrapper(object):
    def __init__(self, model: InferenceModel, intf: SessionInterface, logger: Logger = None):
        """
        Defines a base class for performing inference etc.
        """

        self._model = model
        self.logger = logger or make_base_logger(self._model.name())
        self._intf = intf

    @property
    def model(self) -> InferenceModel:
        return self._model

    def _run(
        self,
        func: Func,
        model: T,
        x: FrameOrArray,
        name: str,
        y: FrameOrArray = None,
        labels: List[str] = None,
        **kwargs,
    ) -> db.TrainingSession:

        model_name = self._model.name()
        if self._model.is_derived:
            msg = f"'{model_name}' inherits from '{self._model.base.name()}' and is thus not trainable!"
            raise NotImplementedError(msg)

        with self._intf.begin_session(model_name, name) as session:
            self.logger.info(f"Starting training of '{model_name}' with '{name}' and using {x.shape[0]} observations")

            res = func(model, x, y=y, **kwargs)
            self.logger.info(f"Successfully finished training '{model_name}' with '{name}'")

            self.logger.info(f"Now trying to serialize '{model_name}' with '{name}'")
            as_bytes = self._model.serialize(res, x)

            session.add_result(as_bytes, self._model.serializer_backend())

            # ===== Save labels ===== #
            session.add_labels(labels)

            # ===== Save meta data ===== #
            meta_data = self._model.add_metadata(res, x=x, y=y, **kwargs)
            session.add_metadatas(meta_data)

            return session.session

    def do_run(self, modkwargs, x: FrameOrArray, name: str, y: FrameOrArray = None, labels: List[str] = None, **kwargs):
        self._run(self._model.fit, self._model.make_model(**modkwargs), x, name=name, y=y, labels=labels, **kwargs)

    def do_update(
        self, old_name: str, x: FrameOrArray, name: str, y: FrameOrArray = None, labels: List[str] = None, **kwargs
    ):
        res = self._run(self._model.update, self.load(old_name), x, name=name, y=y, labels=labels, **kwargs)
        old = self.get_session(old_name)

        # ===== Create link ===== #
        link = db.UpdatedSession(base=old.id, new=res.id)

        self._intf.create(link)

    def do_predict(self, model_name: str, x, orient, as_array=False, **kwargs):
        x_hat = self._model.predict(self.load(model_name), x, **kwargs)

        if isinstance(x_hat, pd.Series):
            x_hat = x_hat.to_frame(x_hat.name or "y")

        if as_array and isinstance(x_hat, pd.DataFrame):
            x_resp = x_hat.values.tolist()
        elif isinstance(x_hat, pd.DataFrame):
            x_resp = x_hat.to_json(orient=orient)
        else:
            x_resp = x_hat.tolist()

        return {
            "data": x_resp,
            "orient": orient,
        }

    def get_session(self, session_name: str, only_succeeded=False) -> Optional[db.TrainingSession]:
        if self._model.is_derived:
            self.logger.info(f"'{self._model.name()}' is derived and loads model from {self._model.base.name()}")
            return ModelWrapper(self._model.base, self._intf).get_session(session_name)

        return self._intf.get_session(self._model.name(), session_name, only_succeeded=only_succeeded)

    def load(self, session_name: str) -> T:
        if self._model.is_derived:
            self.logger.info(f"'{self._model.name()}' is derived and loads model from {self._model.base.name()}")
            return ModelWrapper(self._model.base, self._intf).load(session_name)

        session = self.get_session(session_name, only_succeeded=True)

        if session is None:
            return None

        res = self._intf.get(db.Result, lambda u: u.session_id == session.id, one=True)
        return self._model.deserialize(res.bytes)

    def session_exists(self, session_name: str) -> bool:
        session = self.get_session(session_name)
        return session is not None
