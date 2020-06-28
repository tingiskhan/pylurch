from .base import BaseModelManager
import platform
from ..database import TrainingSession, Model, Base, MetaData
from datetime import datetime
from ..enums import ModelStatus
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from typing import Union


class SQLModelManager(BaseModelManager):
    def __init__(self, session_factory: Union[sessionmaker, scoped_session]):
        """
        TrainingSession manager for SQL based storing.
        :param session_factory: The session factory to use
        """

        super().__init__()
        self._session_maker = session_factory

    def initialize(self):
        Base.metadata.create_all(bind=self._session_maker.bind)

    def make_session(self) -> Session:
        return self._session_maker()

    def close_all_running(self):
        session = self.make_session()

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
        session.close()

        return

    def _persist(self, schema):
        session = self.make_session()

        model = session.query(Model).filter(Model.name == schema['model_name']).one_or_none()
        if not model:
            model = Model(name=schema['model_name'])
            session.add(model)

            model.training_sessions = []

        model.training_sessions += [
            TrainingSession(
                upd_by=schema['upd_by'],
                hash_key=schema['hash_key'],
                start_time=datetime.now(),
                status=schema['status'],
                backend=schema['backend']
            )
        ]

        session.commit()
        session.close()

        return self

    def _update(self, schema):
        s = self.make_session()

        session = s.query(TrainingSession).filter(
            TrainingSession.hash_key == schema['hash_key'],
            TrainingSession.status == ModelStatus.Running,
            TrainingSession.backend == schema['backend'],
            Model.name == schema['model_name'],
            TrainingSession.model_id == Model.id,
        ).one()  # type: TrainingSession

        session.end_time = schema['end_time']
        session.status = schema['status']
        session.byte_string = schema['byte_string']

        if 'meta_data' in schema and len(schema['meta_data']) > 0:
            session.meta_data = [
                MetaData(key=k, value=v) for k, v in schema['meta_data'].items()
            ]

        s.commit()
        s.close()

        return self

    def _get_session(self, name, key, backend, status=None):
        session = self.make_session()

        query = session.query(TrainingSession).filter(
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend,
            Model.name == name,
            TrainingSession.model_id == Model.id,
        )

        if isinstance(status, ModelStatus):
            query = query.filter(TrainingSession.status == status)
        elif status is None:
            ...
        else:
            query = query.filter(TrainingSession.status.in_(status))

        model = query.order_by(TrainingSession.start_time.desc()).first()  # type: TrainingSession

        schema = model.to_schema() if model is not None else None
        session.close()

        return schema

    def delete(self, name, key, backend):
        session = self.make_session()

        deleted = session.query(TrainingSession).filter(
            Model.name == name,
            TrainingSession.model_id == Model.id,
            TrainingSession.hash_key == key,
            TrainingSession.backend == backend
        ).delete(synchronize_session='fetch')

        session.commit()
        session.close()

        return self
