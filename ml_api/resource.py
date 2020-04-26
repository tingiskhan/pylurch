from flask_restful import Resource
from flask_restful.reqparse import RequestParser

# ===== Base parser ===== #
base_parser = RequestParser()

# ===== Argument parser ===== #
arg_parser = base_parser.copy()
arg_parser.add_argument('x', type=str, required=True, help='JSON of data')
arg_parser.add_argument('orient', type=str, help='The orientation of the JSON of the data', required=True)

# ===== Patch parser ===== #
patch_parser = arg_parser.copy()
patch_parser.add_argument('y', type=str, help='The response variable')

# ====== Put parser ====== #
put_parser = patch_parser.copy()
put_parser.add_argument('name', type=str, help='Name of the data set, used as key if provided')
put_parser.add_argument('modkwargs', type=dict, help='Kwargs for model instantiation')
put_parser.add_argument('algkwargs', type=dict, help='Kwargs for algorithm')
put_parser.add_argument('retrain', type=str, help='Whether to retrain using other data', default='False')

# ===== Get parser ====== #
get_parser = base_parser.copy()
get_parser.add_argument('model_key', type=str, required=True, help='Key of the model')

patch_parser.add_argument('model_key', type=str, required=True, help='Key of the model')

# ===== Post parser ====== #
post_parser = patch_parser.copy()


class BaseModelResource(Resource):
    _logger = None
    _executor = None
    _model_manager = None

    @classmethod
    def set_objects(cls, logger, executor, model_manager):
        cls._logger = logger
        cls._executor = executor
        cls._model_manager = model_manager

        return cls

    @property
    def logger(self):
        return self._logger

    @property
    def executor(self):
        return self._executor

    @property
    def model_manager(self):
        return self._model_manager

    def _put(self, **kwargs):
        raise NotImplementedError()

    def _post(self, **kwargs):
        raise NotImplementedError()

    def _patch(self, **kwargs):
        raise NotImplementedError()

    def _delete(self, model_key):
        raise NotImplementedError()

    def _get(self, model_key):
        raise NotImplementedError()

    def auth(self, f):
        """
        Authorization method to decorate API calls with in inherited classes.
        :return: callable
        """

        return f

    def get(self):
        return self.auth(self._get)(**get_parser.parse_args())

    def put(self):
        return self.auth(self._put)(**put_parser.parse_args())

    def post(self):
        return self.auth(self._post)(**post_parser.parse_args())

    def patch(self):
        return self.auth(self._patch)(**put_parser.parse_args())

    def delete(self):
        return self.auth(self._delete)(**get_parser.parse_args())
