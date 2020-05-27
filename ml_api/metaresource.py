from flask_restful import Resource
from .schemas import PostParser, PutParser, GetParser, PatchParser
from flask import request
from .schemas import PostResponse, PutResponse, GetResponse, PatchResponse


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
        """

        return f

    def _apply_and_parse(self, meth, parser, resp):
        res = self.auth(meth)(**parser.load(request.json))
        return resp.dump(res)

    def get(self):
        return self._apply_and_parse(self._get, GetParser(), GetResponse())

    def put(self):
        return self._apply_and_parse(self._put, PutParser(), PutResponse())

    def post(self):
        return self._apply_and_parse(self._post, PostParser(), PostResponse())

    def patch(self):
        return self._apply_and_parse(self._patch, PatchParser(), PatchResponse())

    def delete(self):
        return self._apply_and_parse(self._delete, GetParser(), GetResponse())
