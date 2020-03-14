from .base import BaseModelManager
import platform
from ml_server.db.models import TrainingSession, Model
from datetime import datetime
from ..db.enums import ModelStatus


class SQLModelManager(BaseModelManager):
    def __init__(self, logger, session_maker):
        """
        TrainingSession manager for SQL based storing.
        :param session_maker: The session maker used for connecting to data bases
        :type session_maker: sqlalchemy.orm.sessionmaker
        """

        super().__init__(logger)
        self._sessionmaker = session_maker

    def close_all_running(self):
        session = self._sessionmaker()

        sessions = session.query(TrainingSession).filter(
            TrainingSession.status == ModelStatus.Running,
            TrainingSession.upd_by == platform.node()
        ).all()

        if not sessions:
            return

        self._logger.info(f'Encountered {len(sessions)} running training session, but just started - closing!')

        for s in sessions:
            s.end_time = datetime.now()
            s.status = ModelStatus.Failed

        session.commit()

        return

    def pre_model_start(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(Model.name == name).one_or_none()
        if not model:
            model = Model(name=name)
            session.add(model)

        model.training_sessions = [
            TrainingSession(
                hash_key=key,
                start_time=datetime.now(),
                status=ModelStatus.Running,
                backend=backend
            )
        ]

        session.commit()

        return self

    def model_fail(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(TrainingSession).filter(
            TrainingSession.hash_key == key,
            TrainingSession.status == ModelStatus.Running,
            TrainingSession.backend == backend,
            Model.name == name,
            TrainingSession.model_id == Model.id,
        ).one()  # type: TrainingSession

        model.status = ModelStatus.Failed
        model.end_time = datetime.now()

        session.commit()

        return self

    def save(self, name, key, obj, backend):
        session = self._sessionmaker()

        model = session.query(TrainingSession).filter(
            TrainingSession.hash_key == key,
            TrainingSession.status == ModelStatus.Running,
            TrainingSession.backend == backend,
            Model.name == name,
            TrainingSession.model_id == Model.id,
        ).one()  # type: TrainingSession

        model.status = ModelStatus.Done
        model.byte_string = obj
        model.end_time = datetime.now()

        try:
            session.commit()
        except Exception as e:
            self._logger.exception(f'Something went wrong trying to persist: {name}', e)
            model.status = ModelStatus.Failed
            model.byte_string = None

            session.commit()

        return self

    def _load(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend
        ).order_by(TrainingSession.start_time.desc()).first()  # type: TrainingSession

        if model is None:
            return None

        return model.byte_string

    def delete(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend
        ).delete()

        session.commit()

        return self

    def check_status(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend
        ).order_by(TrainingSession.start_time.desc()).first()  # type: TrainingSession

        if model is not None:
            return model.status

        return None