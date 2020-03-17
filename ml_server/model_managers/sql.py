from .base import BaseModelManager
import platform
from ..db.models import TrainingSession, Model
from datetime import datetime
from ..db.enums import ModelStatus


class SQLModelManager(BaseModelManager):
    def __init__(self, logger, session):
        """
        TrainingSession manager for SQL based storing.
        :param session: The session to use
        :type session: sqlalchemy.orm.Session
        """

        super().__init__(logger)
        self._session = session

    def close_all_running(self):
        sessions = self._session.query(TrainingSession).filter(
            TrainingSession.status == ModelStatus.Running.value,
            TrainingSession.upd_by == platform.node()
        ).all()

        if not sessions:
            return

        self._logger.info(f'Encountered {len(sessions)} running training session, but just started - closing!')

        for s in sessions:
            s.end_time = datetime.now()
            s.status = ModelStatus.Failed.value

        self._session.commit()

        return

    def pre_model_start(self, name, key, backend):
        model = self._session.query(Model).filter(Model.name == name).one_or_none()
        if not model:
            model = Model(name=name)
            self._session.add(model)

            model.training_sessions = []

        model.training_sessions += [
            TrainingSession(
                hash_key=key,
                start_time=datetime.now(),
                status=ModelStatus.Running,
                backend=backend
            )
        ]

        self._session.commit()

        return self

    def model_fail(self, name, key, backend):
        model = self._session.query(TrainingSession).filter(
            TrainingSession.hash_key == key,
            TrainingSession.status == ModelStatus.Running.value,
            TrainingSession.backend == backend.value,
            Model.name == name,
            TrainingSession.model_id == Model.id,
        ).one()  # type: TrainingSession

        model.status = ModelStatus.Failed.value
        model.end_time = datetime.now()

        self._session.commit()

        return self

    def save(self, name, key, obj, backend):
        model = self._session.query(TrainingSession).filter(
            TrainingSession.hash_key == key,
            TrainingSession.status == ModelStatus.Running.value,
            TrainingSession.backend == backend.value,
            Model.name == name,
            TrainingSession.model_id == Model.id,
        ).one()  # type: TrainingSession

        model.status = ModelStatus.Done.value
        model.byte_string = obj
        model.end_time = datetime.now()

        try:
            self._session.commit()
        except Exception as e:
            self._logger.exception(f'Something went wrong trying to persist: {name}', e)
            model.status = ModelStatus.Failed.value
            model.byte_string = None

            self._session.commit()

        return self

    def _load(self, name, key, backend):
        model = self._session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend.value
        ).order_by(TrainingSession.start_time.desc()).first()  # type: TrainingSession

        if model is None:
            return None

        return model.byte_string

    def delete(self, name, key, backend):
        model = self._session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend.value
        ).delete()

        self._session.commit()

        return self

    def check_status(self, name, key, backend):
        model = self._session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend.value
        ).order_by(TrainingSession.start_time.desc()).first()  # type: TrainingSession

        if model is not None:
            return model.status

        return None