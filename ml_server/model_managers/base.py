import onnxruntime as rt
from ..db.enums import SerializerBackend
import dill


# TODO: Perhaps use YAML as base to streamline?
class BaseModelManager(object):
    def __init__(self, logger):
        """
        Defines a base class for model management.
        :param logger: The logger
        """

        self._logger = logger

    def close_all_running(self):
        """
        Closes all running tasks for this node. Called when starting the app to avoid having "broken" models.
        """

        raise NotImplementedError()

    def pre_model_start(self, name, key, backend):
        """
        What to do before starting model. E.g. create a database entry or a YAML-file.
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
        What to do on model fail.
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
        Save the model.
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
        Method for deleting model.
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