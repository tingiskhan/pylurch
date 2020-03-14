import onnxruntime as rt
from ml_server.db.enums import SerializerBackend
import dill


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