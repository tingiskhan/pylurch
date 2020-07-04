from ..enums import SerializerBackend, ModelStatus
import platform
from datetime import datetime
from ..schemas import ModelSchema
from logging import Logger
from typing import Dict


class BaseModelManager(object):
    def __init__(self):
        """
        Defines a base class for model management.
        """

        self._logger = None
        self._ms = ModelSchema()

    def initialize(self):
        """
        Initializes the model manager.
        """

        return self

    def set_logger(self, logger: Logger):
        self._logger = logger

    def close_all_running(self):
        """
        Closes all running tasks for this node. Called when starting the app to avoid having "broken" models.
        """

        raise NotImplementedError()

    def pre_model_start(self, name: str, key: str, backend: SerializerBackend):
        """
        What to do before starting model. E.g. create a database entry or a YAML-file.
        :param name: The name of the model
        :param key: The key
        :param backend: The backend to use
        """

        asdict = {
            'upd_by': platform.node(),
            'model_name': name,
            'hash_key': key,
            'status': ModelStatus.Running,
            'start_time': datetime.now(),
            'backend': backend
        }

        self._persist(asdict)

        return self

    def model_fail(self, name: str, key: str, backend: SerializerBackend):
        """
        What to do on model fail.
        :param name: Name of the session
        :param key: The key
        :param backend: The backend to use
        :type backend: SerializerBackend
        """

        data = self._get_session(name, key, backend, status=ModelStatus.Running)

        data['status'] = ModelStatus.Failed
        data['end_time'] = datetime.now()

        self._update(data)

        return self

    def _get_session(self, name: str, key: str, backend: SerializerBackend,
                     status: ModelStatus = None) -> Dict[str, object]:
        """
        Get the given session of model with `name` by ways of `key`. Returns the session represented as a dictionary
        :param name: The name of the model
        :param key: The hash key
        :param backend: The backend
        :param status: If to filter on status
        """

        raise NotImplementedError()

    def _persist(self, schema: dict):
        """
        Persist the data for the first time.
        :param schema: The schema to persist.
         Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()

    def _update(self, schema: dict):
        """
        Update an existing entry.
        :param schema: The schema to persist.
         Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()

    def check_status(self, name: str, key: str, backend: SerializerBackend) -> ModelStatus:
        """
        Checks the status.
        :param name: The name of the model
        :param key: The key of the model
        :param backend: The backend to use
         String indicating status
        """

        schema = self._get_session(name, key, backend)

        if schema is None:
            return ModelStatus.Unknown

        return schema['status']

    def get(self, name: str, key: str, backend: SerializerBackend, status: ModelStatus) -> Dict[str, object]:
        """
        Get the given session of model with `name` by ways of `key`. Returns the session represented as a dictionary
        :param name: The name of the model
        :param key: The hash key
        :param backend: The backend
        :param status: If to filter on status
        """

        return self._get_session(name, key, backend, status)

    def load(self, name: str, key: str, backend: SerializerBackend) -> bytes or None:
        """
        Loads the model.
        :param name: The name of the model to save
        :param key: The data key
        :param backend: The backend to use
         The byte string
        """

        saved_model = self._get_session(name, key, backend, status=ModelStatus.Done)

        if saved_model is None:
            return None

        return saved_model['byte_string']

    def save(self, name: str, key: str, obj: object, backend: SerializerBackend, meta_data: Dict[str, str] = None):
        """
        Save the model.
        :param name: The name of the model to save
        :param key: The key of the data
        :param obj: The model to save in byte string
        :param backend: The backend to use
        :param meta_data: Meta data to add to the session
         Self
        :rtype: BaseModelManager
        """

        ms = self._get_session(name, key, backend, status=ModelStatus.Running)

        ms['status'] = ModelStatus.Done
        ms['end_time'] = datetime.now()
        ms['byte_string'] = obj

        if meta_data is not None:
            ms['meta_data'] = meta_data

        self._update(ms)

        return self

    def delete(self, name: str, key: str, backend: SerializerBackend):
        """
        Method for deleting model.
        :param name: The name of the model
        :param key: The key of the model to save
        :param backend: The backend to use
         Self
        :rtype: BaseModelManager
        """

        raise NotImplementedError()
