import os
from abc import ABC
import onnxruntime as rt
import platform
from .db.models import TrainingSession, Model
from datetime import datetime
import glob
from .db.enums import ModelStatus, SerializerBackend
import dill
import yaml


class BaseModelManager(object):
    def __init__(self, logger):
        """
        Defines a base class for model management.
        :param logger: The logger
        """

        self._logger = logger

    def close_all_running(self):
        """
        Closes all running tasks. Mainly for use when starting.
        """

        raise NotImplementedError()

    def pre_model_start(self, name, key, backend):
        """
        If to do anything prior the starting the model.
        :param name: The name of the model
        :type name: str
        :param key: The key
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def model_fail(self, name, key, backend):
        """
        What to do on model fail. Usually nothing as data haven't been saved. Used in database scenarios
        :param name: The name of the model
        :type name: str
        :param key: The key
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def check_status(self, name, key, backend):
        """
        Checks the status.
        :param name: The name of the model
        :type name: str
        :param key: The key of the model
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: String indicating status
        :rtype: str
        """

        raise NotImplementedError()

    def load(self, name, key, backend):
        """
        Loads the model.
        :param name: The name of the model to save
        :type name: str
        :param key: The data key
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: onnxruntime.InferenceSession
        """

        bytestring = self._load(name, key, backend)

        if bytestring is None:
            return None

        if backend == SerializerBackend.ONNX:
            return rt.InferenceSession(bytestring)
        elif backend == SerializerBackend.Dill:
            return dill.loads(bytestring)

    def save(self, name, key, obj, backend):
        """
        Saving the model.
        :param name: The name of the model to save
        :type name: str
        :param key: The key of the data
        :type key: str
        :param obj: The model to save in byte string
        :type obj: bytes
        :param backend: The backend to use
        :type backend: str
        :return: None
        :rtype: None
        """

        raise NotImplementedError()

    def delete(self, name, key, backend):
        """
        Method for deleting object.
        :param name: The name of the model
        :type name: str
        :param key: The key of the model to save
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()

    def _load(self, name, key, backend):
        raise NotImplementedError()


class FileModelManager(BaseModelManager, ABC):
    def __init__(self, logger, prefix, **kwargs):
        """
        Defines a base class for model management.
        :param prefix: The prefix for each model file
        :type prefix: str
        """

        super().__init__(logger, **kwargs)

        self._pref = prefix

    def close_all_running(self):
        ymls = glob.glob(f'{self._pref}/*.yml', recursive=True)

        running = 0
        for f in ymls:
            yml = self._get_yml(f)

            if yml.get('platform') != platform.node():
                continue
            if yml['status'] != ModelStatus.Running:
                continue

            yml['end-time'] = datetime.now()
            yml['status'] = ModelStatus.Failed

            self._save_yml(f, yml)

            running += 1

        if running > 0:
            self._logger.info(f'Encountered {running} running training session, but just started - closing!')

        return

    @staticmethod
    def _get_ext(backend):
        return 'onnx' if backend == SerializerBackend.ONNX else 'pkl'

    def _format_name(self, name, key, backend, yml=False):
        first = f'{name}-{key}'

        if not yml:
            return f'{first}.{self._get_ext(backend)}'

        return f'{first}-{backend}.yml'

    def pre_model_start(self, name, key, backend):
        yml = self._make_yaml(name, key, backend)

        yml['start-time'] = datetime.now()
        yml['end-time'] = datetime.max
        yml['status'] = ModelStatus.Running

        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        self._save_yml(yml_name, yml)

        return self

    def model_fail(self, name, key, backend):
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        yml['end-time'] = datetime.now()
        yml['status'] = ModelStatus.Failed

        return self

    def save(self, name, key, obj, backend):
        # ===== YAML ===== #
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        yml['end-time'] = datetime.now()
        yml['status'] = ModelStatus.Done

        self._save_yml(yml_name, yml)

        # ====== Model ===== #
        name = f'{self._pref}/{self._format_name(name, key, backend)}'
        self._save(name, obj)

        return self

    def check_status(self, name, key, backend):
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        if yml is None:
            return None

        return yml['status']

    def delete(self, name, key, backend):
        raise NotImplementedError()

    def _get_yml(self, path):
        raise NotImplementedError()

    def _save_yml(self, path, yml):
        raise NotImplementedError()

    def _save(self, name, obj):
        raise NotImplementedError()

    def _make_yaml(self, name, key, backend):
        data = {
            'model-name': name,
            'hash-key': key,
            'backend': backend,
            'platform': platform.node()
        }

        return data


class DebugModelManager(FileModelManager):
    def __init__(self, *args, **kwargs):
        """
        Defines a model manager for use in debug settings.
        """

        super().__init__(*args, **kwargs)

        if not os.path.exists(self._pref):
            os.mkdir(self._pref)

    def _save_yml(self, path, yml):
        with open(path, 'w') as f:
            yaml.dump(yml, f)

    def _get_yml(self, path):
        with open(path, 'r') as s:
            return yaml.safe_load(s)

    def _save(self, name, obj):
        with open(name, 'wb') as f:
            f.write(obj)

        return

    def _load(self, name, key, backend):
        name = f'{self._pref}/{self._format_name(name, key, backend)}'

        if not os.path.exists(name):
            return None

        with open(name, 'rb') as f:
            return f.read()

    def delete(self, name, key, backend):
        f = glob.glob(f'{self._pref}/*{self._format_name(name, key, backend)}', recursive=True)

        if len(f) > 1:
            raise ValueError('Multiple models with same name!')

        os.remove(f[-1])

        return self

    def check_status(self, name, key, backend):
        exists = os.path.exists(f'{self._pref}/{self._format_name(name, key, backend)}')

        if exists:
            return ModelStatus.Done

        return None


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
