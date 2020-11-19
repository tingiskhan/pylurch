import pandas as pd
from typing import Union, Callable, Any, Dict, Optional, List
from logging import Logger
import numpy as np
from pylurch.contract.interfaces import DatabaseInterface
from pylurch.contract import database as db, enums as e
from ...utils import make_base_logger
from .model import InferenceModel, T


FrameOrArray = Union[pd.DataFrame, np.ndarray]
Func = Callable[[T, FrameOrArray, Optional[FrameOrArray], Dict[str, Any]], Any]


class ModelWrapper(object):
    def __init__(self, model: InferenceModel, intf: DatabaseInterface, logger: Logger = None):
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
        task_obj: db.Task,
        y: FrameOrArray = None,
        labels: List[str] = None,
        **kwargs,
    ) -> db.TrainingSession:

        modname = self._model.name()
        if self._model.is_derived:
            msg = f"'{modname}' inherits from '{self._model.base.name()}' and is thus not trainable!"
            raise NotImplementedError(msg)

        db_model = self._get_model()
        self.logger.info(f"Starting training of '{modname}' with '{name}' and using {x.shape[0]} observations")

        # ===== Get latest sessions ===== #
        latest = self.get_session(name, only_succeeded=False)

        # ===== Save session ===== #
        session = db.TrainingSession(
            model_id=db_model.id,
            name=name,
            backend=self._model.serializer_backend(),
            task_id=task_obj.id,
            status=e.Status.Unknown,
            version=1 if latest is None else (latest.version + 1),
        )

        session = self._intf.create(session)

        # ===== Fit/update ===== #
        res = func(model, x, y=y, **kwargs)
        self.logger.info(f"Successfully finished training '{modname}' with '{name}'")

        # ===== Serialize model ===== #
        self.logger.info(f"Now trying to serialize '{modname}' with '{name}'")
        as_bytes = self._model.serialize(res, x)

        # ==== Save model ===== #
        data = db.TrainingResult(session_id=session.id, bytes=as_bytes)

        self.logger.info(f"Now trying to persist '{modname}' with '{name}'")
        self._intf.create(data)

        self.logger.info(f"Successfully persisted '{self._model.name()}' with '{name}'")

        # ===== Save labels ===== #
        labels = [db.Label(session_id=session.id, label=lab) for lab in labels]
        if any(labels):
            self._intf.create(labels)

        # ===== Save meta data ===== #
        meta_data = self._model.add_metadata(res, x=x, y=y, **kwargs)
        md = [db.MetaData(session_id=session.id, key=k, value=v) for k, v in meta_data.items()]

        if any(md):
            self._intf.create(md)

        return session

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
        x_hat = self._model.predict(self.load(model_name), self._model.parse_data(x, orient=orient), **kwargs)

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

    def _get_model(self) -> db.Model:
        model = self._intf.get(db.Model, lambda u: u.name == self._model.name(), one=True)

        if model is None:
            model = self._intf.create(db.Model(name=self._model.name()))

        return model

    def get_session(self, name: str, only_succeeded=False) -> Optional[db.TrainingSession]:
        modname = self._model.name()
        if self._model.is_derived:
            self.logger.info(f"'{modname}' is derived and loads model from {self._model.base.name()}")
            return ModelWrapper(self._model.base, self._intf).get_session(name)

        mod = self._intf.get(db.Model, lambda u: u.name == modname, one=True)
        if mod is None:
            return None

        def f(u: db.TrainingSession):
            if only_succeeded:
                return (u.name == name) & (u.model_id == mod.id) & (u.status == e.Status.Done)

            return (u.name == name) & (u.model_id == mod.id)

        return self._intf.get(db.TrainingSession, f, latest=True)

    def load(self, session_name: str) -> T:
        """
        Method for loading the latest successfully trained model with 'session_name'.
        """

        if self._model.is_derived:
            self.logger.info(f"'{self._model.name()}' is derived and loads model from {self._model.base.name()}")
            return ModelWrapper(self._model.base, self._intf).load(session_name)

        session = self.get_session(session_name)

        if session is None:
            return None

        res = self._intf.get(db.TrainingResult, lambda u: u.session_id == session.id, one=True)
        return self._model.deserialize(res.bytes)

    def session_exists(self, session_name: str) -> bool:
        session = self.get_session(session_name)
        return session is not None
