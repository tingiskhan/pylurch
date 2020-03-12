import os
from abc import ABC
import onnxruntime as rt
from google.cloud import storage
from .db.models import Model
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

    def pre_model_start(self, key, backend):
        """
        If to do anything prior the starting the model.
        :param key: The key
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def model_fail(self, key, backend):
        """
        What to do on model fail. Usually nothing as data haven't been saved. Used in database scenarios
        :param key: The key
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def check_status(self, key, backend):
        """
        Checks the status.
        :param key: The key of the model
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: String indicating status
        :rtype: str
        """

        raise NotImplementedError()

    def load(self, name, backend):
        """
        Loads the model.
        :param name: The name of the model to save
        :type name: str
        :param backend: The backend to use
        :type backend: str
        :return: onnxruntime.InferenceSession
        """

        bytestring = self._load(name, backend)

        if bytestring is None:
            return None

        if backend == SerializerBackend.ONNX:
            return rt.InferenceSession(bytestring)
        elif backend == SerializerBackend.Dill:
            return dill.loads(bytestring)

    def save(self, name, obj, backend):
        """
        Saving the model.
        :param name: The name of the model to save
        :type name: str
        :param obj: The model to save in byte string
        :type obj: bytes
        :param backend: The backend to use
        :type backend: str
        :return: None
        :rtype: None
        """

        raise NotImplementedError()

    def delete(self, key, backend):
        """
        Method for deleting object.
        :param key: The name of the model to save
        :type key: str
        :param backend: The backend to use
        :type backend: str
        :return: Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()

    def _load(self, name, backend):
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

    def _format_name(self, name, backend):
        return f'{name}.{self._get_ext(backend)}'

    def save(self, name, obj, backend):
        name = f'{self._pref}/{self._format_name(name, backend)}'
        self._save(name, obj)

        return self

    def delete(self, key, backend):
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

    def _load(self, name, backend):
        name = f'{self._pref}/{name}.{self._get_ext(backend)}'

        if not os.path.exists(name):
            return None

        with open(name, 'rb') as f:
            return f.readlines()

    def delete(self, key, backend):
        f = glob.glob(f'{self._pref}/*{key}.{self._get_ext(backend)}', recursive=True)

        if len(f) > 1:
            raise ValueError('Multiple models with same name!')

        os.remove(f[-1])

        return self

    def check_status(self, key, backend):
        exists = os.path.exists(f'{self._pref}/{self._format_name(key, backend)}')

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

    def _load(self, name, backend):
        name = f'{self._pref}/{name}.{self._get_ext(backend)}'

        client = storage.Client()
        bucket = client.get_bucket(self._bucket)
        blob = bucket.get_blob(name)

        if blob is None or not blob.exists():
            return None

        return blob.download_as_string()

    def _get_blob(self, key, backend):
        client = storage.Client()

        blobs = client.list_blobs(self._bucket, prefix=f'{self._pref}')

        return (blob for blob in blobs if blob.name.endswith(f'{self._format_name(key, backend)}'))

    def delete(self, key, backend):
        for blob in self._get_blob(key, backend):
            blob.delete()

        return self

    def check_status(self, key, backend):
        blob_exists = len(self._get_blob(key, backend)) > 0

        if blob_exists:
            return ModelStatus.Done

        return None


class SQLModelManager(BaseModelManager):
    def __init__(self, logger, session_maker):
        """
        Model manager for SQL based storing.
        :param session_maker: The session maker used for connecting to data bases
        :type session_maker: sqlalchemy.orm.sessionmaker
        :param kwargs:
        """

        super().__init__(logger)
        self._sessionmaker = session_maker

    def pre_model_start(self, key, backend):
        model_run = Model(
            hash_key=key,
            start_time=datetime.now(),
            status=ModelStatus.Running,
            backend=backend
        )

        session = self._sessionmaker()
        session.add(model_run)
        session.commit()

        return self

    def model_fail(self, key, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key,
            Model.status == ModelStatus.Running,
            Model.backend == backend
        ).one()  # type: Model

        model.status = ModelStatus.Failed
        model.end_time = datetime.now()

        session.commit()

        return self

    def save(self, name, obj, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == name,
            Model.status == ModelStatus.Running,
            Model.backend == backend
        ).one()  # type: Model

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

    def _load(self, name, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == name,
            Model.backend == backend
        ).order_by(Model.start_time.desc()).first()  # type: Model

        if model is None:
            return None

        return model.byte_string

    def delete(self, key, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key,
            Model.backend == backend
        ).delete()

        session.commit()

        return self

    def check_status(self, key, backend):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key,
            Model.backend == backend
        ).order_by(Model.start_time.desc()).first()  # type: Model

        if model is not None:
            return model.status

        return None
