import os
from abc import ABC

import onnxruntime as rt
from google.cloud import storage
from ml_server.db.models import Model
from datetime import datetime
import glob
from .db.enums import ModelStatus


class BaseModelManager(object):
    def __init__(self, logger):
        """
        Defines a base class for model management.
        """

        self._logger = logger

    def pre_model_start(self, key):
        """
        If to do anything prior the starting the model.
        :param key: The key
        :type key: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def model_fail(self, key):
        """
        What to do on model fail. Usually nothing as data haven't been saved. Used in database scenarios
        :param key: The key
        :type key: str
        :return: Self
        :rtype: BaseModelManager
        """

        return self

    def check_status(self, key):
        """
        Checks the status.
        :param key: The key of the model
        :type key: str
        :return: String indicating status
        :rtype: str
        """

        raise NotImplementedError()

    def load(self, name):
        """
        Loads the model.
        :param name: The name of the model to save
        :type name: str
        :return: onnxruntime.InferenceSession
        """

        bytestring = self._load(name)

        if bytestring is None:
            return None

        return rt.InferenceSession(bytestring)

    def save(self, name, obj):
        """
        Saving the model.
        :param name: The name of the model to save
        :type name: str
        :param obj: The model to save in byte string
        :type obj: bytes
        :return: None
        :rtype: None
        """

        raise NotImplementedError()

    def delete(self, key):
        """
        Method for deleting object.
        :param key: The name of the model to save
        :type key: str
        :return: Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()

    def _load(self, name):
        raise NotImplementedError()


class FileModelManager(BaseModelManager, ABC):
    def __init__(self, logger, prefix, ext='pkl'):
        """
        Defines a base class for model management.
        :param prefix: The prefix for each model file
        :type prefix: str
        """

        super().__init__(logger)

        self._pref = prefix
        self._ext = ext

    def _format_name(self, name):
        return f'{name}.{self._ext}'

    def save(self, name, obj):
        name = f'{self._pref}/{self._format_name(name)}'
        self._save(name, obj)

        return self

    def delete(self, key):
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

    def _load(self, name):
        name = f'{self._pref}/{name}.{self._ext}'

        if not os.path.exists(name):
            return None

        with open(name, 'rb') as f:
            return f.readlines()

    def delete(self, key):
        f = glob.glob(f'{self._pref}/*{key}.{self._ext}', recursive=True)

        if len(f) > 1:
            raise ValueError('Multiple models with same name!')

        os.remove(f[-1])

        return self

    def check_status(self, key):
        exists = os.path.exists(f'{self._pref}/{self._format_name(key)}')

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

    def _load(self, name):
        name = f'{self._pref}/{name}.{self._ext}'

        client = storage.Client()
        bucket = client.get_bucket(self._bucket)
        blob = bucket.get_blob(name)

        if blob is None or not blob.exists():
            return None

        return blob.download_as_string()

    def _get_blob(self, key):
        client = storage.Client()

        blobs = client.list_blobs(self._bucket, prefix=f'{self._pref}')

        return (blob for blob in blobs if blob.name.endswith(f'{self._format_name(key)}'))

    def delete(self, key):
        for blob in self._get_blob(key):
            blob.delete()

        return self

    def check_status(self, key):
        blob_exists = len(self._get_blob(key)) > 0

        if blob_exists:
            return ModelStatus.Done

        return None


class SQLModelManager(BaseModelManager):
    def __init__(self, logger, session_maker):
        super().__init__(logger)
        self._sessionmaker = session_maker

    def pre_model_start(self, key):
        model_run = Model(
            hash_key=key,
            start_time=datetime.now(),
            status=ModelStatus.Running,
        )

        session = self._sessionmaker()
        session.add(model_run)
        session.commit()

        return self

    def model_fail(self, key):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key,
            Model.status == ModelStatus.Running
        ).one()  # type: Model

        model.status = ModelStatus.Failed
        model.end_time = datetime.now()

        session.commit()

        return self

    def save(self, name, obj):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == name,
            Model.status == ModelStatus.Running
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

    def _load(self, name):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == name
        ).order_by(Model.start_time.desc()).first()  # type: Model

        if model is None:
            return None

        return model.byte_string

    def delete(self, key):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key
        ).delete()

        session.commit()

        return self

    def check_status(self, key):
        session = self._sessionmaker()

        model = session.query(Model).filter(
            Model.hash_key == key,
        ).order_by(Model.start_time.desc()).first()  # type: Model

        if model is not None:
            return model.status

        return None
