import pandas as pd
from .utils import custom_error
from hashlib import sha256
from .enums import ModelStatus
from typing import Dict
from .inference import InferenceModel
from .schemas import PostParser, PutParser, GetParser, PatchParser
from .schemas import PostResponse, PutResponse, GetResponse, PatchResponse


class ModelResource(object):
    def __init__(self, model_resource: InferenceModel, queue_meth):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param queue_meth:
        """

        self.model_resource = model_resource
        self.queue_meth = queue_meth

    @property
    def logger(self):
        return self.model_resource.logger

    def _apply_and_parse(self, meth, request, res, parser, resp):
        if meth in (self._get, self._delete):
            temp = meth(**parser.load(request.params))
        else:
            temp = meth(**parser.load(request.media))

        res.media = resp.dump(temp)

        return res

    def on_get(self, req, res):
        return self._apply_and_parse(self._get, req, res, GetParser(), GetResponse())

    def on_put(self, req, res):
        return self._apply_and_parse(self._put, req, res, PutParser(), PutResponse())

    def on_post(self, req, res):
        return self._apply_and_parse(self._post, req, res, PostParser(), PostResponse())

    def on_patch(self, req, res):
        return self._apply_and_parse(self._patch, req, res, PatchParser(), PatchResponse())

    def on_delete(self, req, res):
        return self._apply_and_parse(self._delete, req, res, GetParser(), GetResponse())

    def parse_data(self, data: str, **kwargs) -> pd.DataFrame:
        """
        Method for parsing data.
        :param data: The data in string format
         Data in required format
        """

        return pd.read_json(data, **kwargs).sort_index()

    @custom_error
    def _put(self, x: str, orient: str, name: str, y: str = None, modkwargs: Dict[str, object] = None,
             algkwargs: Dict[str, object] = None, retrain: bool = False):
        # ===== Get data ===== #
        x = self.parse_data(x, orient=orient)

        # ===== Generate model key ===== #
        key = sha256(name.lower().encode()).hexdigest()

        # ===== Check status ===== #
        status = self.model_resource.check_status(key)

        if status not in (ModelStatus.Unknown, ModelStatus.Failed) and not retrain:
            self.logger.info('Model already exists, and no retrain requested')
            return {'status': status, 'model_key': key}

        # ===== Define model ===== #
        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()

        if y is not None:
            akws['y'] = self.parse_data(y, orient=orient)

        model = self.model_resource.make_model(**modkwargs)

        if status == ModelStatus.Running:
            if retrain:
                return {'status': status, 'model_key': key}, 400

            return {'status': status, 'model_key': key}

        # ===== Let it persist run first ===== #
        self.model_resource.pre_model_start(key)

        # ===== Start background task ===== #
        self.queue_meth(key, self.model_resource.do_run, model, x, key, **akws)

        return {'status': self.model_resource.check_status(key), 'model_key': key}

    @custom_error
    def _post(self, model_key: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, object]):
        status = self.model_resource.check_status(model_key)

        if status != ModelStatus.Done:
            return {'data': None, 'orient': orient}, 400

        mod = self.model_resource.load(model_key)

        if mod is None:
            return {'data': None, 'orient': orient}, 400

        self.logger.info(f'Predicting values using model {self.model_resource.name()}')

        x_hat = self.model_resource.predict(mod, self.parse_data(x, orient=orient), **kwargs)

        if as_array:
            x_resp = x_hat.values.tolist()
        else:
            x_resp = x_hat.to_json(orient=orient)

        resp = {
            'data': x_resp,
            'orient': orient,
        }

        return resp

    @custom_error
    def _get(self, model_key):
        status = self.model_resource.check_status(model_key)

        return {'status': status}

    @custom_error
    def _patch(self, model_key: str, x: str, orient: str, y: str = None):
        status = self.model_resource.check_status(model_key)

        if status != ModelStatus.Done:
            return {'status': status}

        model = self.model_resource.load(model_key)
        x = self.parse_data(x, orient=orient)

        kwargs = dict()
        if y is not None:
            kwargs['y'] = self.parse_data(y, orient=orient)

        # ===== Let it persist run first ===== #
        self.model_resource.pre_model_start(model_key)

        # ===== Start background task ===== #
        self.queue_meth(model_key, self.model_resource.do_update, model, x, model_key)

        return {'status': self.model_resource.check_status(model_key)}

    @custom_error
    def _delete(self, model_key):
        self.logger.info(f'Deleting model with key: {model_key}')

        try:
            self.model_resource.delete(model_key)
            self.logger.info(f'Successfully deleted model with key: {model_key}')
        except Exception as e:
            self.logger.exception(f'Failed to delete object with key {model_key}', e)
            return {'status': ModelStatus.Failed}

        return {'status': ModelStatus.Done}
