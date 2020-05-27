from flask_restful import Api
from .metaresource import BaseModelResource
from flask_executor import Executor
from .model_managers import BaseModelManager


class MachineLearningApi(Api):
    def __init__(self, model_manager: BaseModelManager, **kwargs):
        """
        An extension of flask_restful.Api for facilitating the exposing of Machine Learning models.
        :param model_manager: The model manager
        """

        self._executor = None
        self._model_manager = model_manager
        self._logger = None

        super().__init__(**kwargs)

    # NOTE: this is the exact same code as flask_restful, only with the addition that we define executor and logger
    def init_app(self, app):
        self._executor = Executor(app)
        self._logger = app.logger

        # ===== Model manager related ====== #
        self._model_manager.initialize()
        self._model_manager.set_logger(self._logger)
        self._model_manager.close_all_running()

        try:
            app.record(self._deferred_blueprint_init)
        except AttributeError:
            self._init_app(app)
        else:
            self.blueprint = app

    # NOTE: this is the exact same code as flask_restful, only with the addition that we set static variables
    def add_resource(self, resource, *urls, **kwargs):
        if issubclass(resource, BaseModelResource):
            resource = resource.set_objects(self._logger, self._executor, self._model_manager)

        if self.app is not None:
            self._register_view(self.app, resource, *urls, **kwargs)
        else:
            self.resources.append((resource, urls, kwargs))