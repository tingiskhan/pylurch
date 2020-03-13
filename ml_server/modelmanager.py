import os
from abc import ABC
import onnxruntime as rt
from google.cloud import storage
from .db.models import TrainingSession, Model
from datetime import datetime
import glob
from .db.enums import ModelStatus, SerializerBackend
import dill


class BaseModelManager(object):
    def __init__(self, logger):
        """
        Defines a base class for model management.
        :param logger: The logger
        """

        self._logger = logger

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

    @staticmethod
    def _get_ext(backend):
        return 'onnx' if backend == SerializerBackend.ONNX else 'pkl'

    def _format_name(self, name, key, backend):
        return f'{name}-{key}.{self._get_ext(backend)}'

    def save(self, name, key, obj, backend):
        name = f'{self._pref}/{self._format_name(name, key, backend)}'
        self._save(name, obj)

        return self

    def delete(self, name, key, backend):
        raise NotImplementedError()

    def _save(self, name, obj):
        raise NotImplementedError()


class DebugModelManager(FileModelManager):
    def __init__(self, *args, **kwargs):
        """
        Defines a model manager for use in debug settings.
        """

        super().__init__(*args, **kwargs)

        if not os.path.exists(self._pref):
            os.mkdir(self._pref)

    def _save(self, name, obj):
        with open(name, 'wb') as f:
            f.write(obj)

        return

    def _load(self, name, key, backend):
        name = f'{self._pref}/{self._format_name(name, key, backend)}'

        if not os.path.exists(name):
            return None

        with open(name, 'rb') as f:
            return f.readlines()

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


class GoogleCloudStorage(FileModelManager):
    def __init__(self, logger, bucket, *args, **kwargs):
        """
        Defines a model manager for use when Google cloud is backend.
        :param bucket: The bucket to use
        :type bucket: str
        """
        super().__init__(logger, *args, **kwargs)
        self._bucket = bucket

        self._verify_bucket_exists()

    def _verify_bucket_exists(self):
        client = storage.Client()

        if not client.get_bucket(self._bucket):
            client.create_bucket(self._bucket)

        # ===== Create a small file to verify it works ===== #
        bucket = client.get_bucket(self._bucket)
        blob = bucket.blob('this-is-a-test.txt')

        blob.upload_from_string('Ha! I can upload')

    def _save(self, name, obj):
        client = storage.Client()
        bucket = client.get_bucket(self._bucket)

        blob = bucket.blob(name)

        blob.upload_from_string(obj)

    def _load(self, name, key, backend):
        name = f'{self._pref}/{self._format_name(name, key, backend)}'

        client = storage.Client()
        bucket = client.get_bucket(self._bucket)
        blob = bucket.get_blob(name)

        if blob is None or not blob.exists():
            return None

        return blob.download_as_string()

    def _get_blob(self, name, key, backend):
        client = storage.Client()

        blobs = client.list_blobs(self._bucket, prefix=f'{self._pref}')

        return (blob for blob in blobs if blob.name.endswith(f'{self._format_name(name, key, backend)}'))

    def delete(self, name, key, backend):
        for blob in self._get_blob(name, key, backend):
            blob.delete()

        return self

    def check_status(self, name, key, backend):
        blob_exists = len(self._get_blob(name, key, backend)) > 0

        if blob_exists:
            return ModelStatus.Done

        return None


class SQLModelManager(BaseModelManager):
    def __init__(self, logger, session_maker):
        """
        TrainingSession manager for SQL based storing.
        :param session_maker: The session maker used for connecting to data bases
        :type session_maker: sqlalchemy.orm.sessionmaker
        :param kwargs:
        """

        super().__init__(logger)
        self._sessionmaker = session_maker

    def pre_model_start(self, name, key, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(Model.name == name).one_or_none()
        if not model:
            model = Model(name=name)
            session.add(model)

        model.training_sessions = [TrainingSession(
            hash_key=key,
            start_time=datetime.now(),
            status=ModelStatus.Running,
            backend=backend
        )]

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
