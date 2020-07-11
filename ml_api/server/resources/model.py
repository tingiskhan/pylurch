import pandas as pd
from ml_api.utils import custom_error
from hashlib import sha256
from ml_api.contract.enums import ModelStatus
from typing import Dict
from ml_api.server.inference import InferenceModel
import ml_api.contract.schemas as sc
from falcon.status_codes import HTTP_200, HTTP_400


class ModelResource(object):
    def __init__(self, model_resource: InferenceModel, queue_meth):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param queue_meth: The method used for queuing tasks
        """

        self.model_resource = model_resource
        self.queue_meth = queue_meth

    @property
    def logger(self):
        return self.model_resource.logger

    def _apply_and_parse(self, meth, request, res, parser, resp):
        if meth == self._get:
            temp, status = meth(**parser.load(request.params))
        else:
            temp, status = meth(**parser.load(request.media))

        res.media = resp.dump(temp)
        res.status = status

        return res

    def on_get(self, req, res):
        return self._apply_and_parse(self._get, req, res, sc.GetParser(), sc.GetResponse())

    def on_put(self, req, res):
        return self._apply_and_parse(self._put, req, res, sc.PutParser(), sc.PutResponse())

    def on_post(self, req, res):
        return self._apply_and_parse(self._post, req, res, sc.PostParser(), sc.PostResponse())

    def on_patch(self, req, res):
        return self._apply_and_parse(self._patch, req, res, sc.PatchParser(), sc.PatchResponse())

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
            self.logger.info(f"Instance '{key}' of '{self.model_resource.name()}' already exists, no retrain requested")
            return {'status': status, 'model_key': key}, HTTP_200

        # ===== Define model ===== #
        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()

        if y is not None:
            akws['y'] = self.parse_data(y, orient=orient)

        model = self.model_resource.make_model(**modkwargs)

        if status == ModelStatus.Running:
            if retrain:
                return {'status': status, 'model_key': key}, HTTP_200

            return {'status': status, 'model_key': key}, HTTP_200

        # ===== Let it persist run first ===== #
        session = self.model_resource.initialize_training(key)

        # ===== Start background task ===== #
        self.queue_meth(key, self.model_resource.do_run, model, x, session, **akws)

        return {'status': self.model_resource.check_status(key), 'model_key': key}, HTTP_200

    @custom_error
    def _post(self, model_key: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, object]):
        status = self.model_resource.check_status(model_key)

        if status != ModelStatus.Done:
            return {'data': None, 'orient': orient}, HTTP_400

        mod = self.model_resource.load(model_key)

        if mod is None:
            return {'data': None, 'orient': orient}, HTTP_400

        self.logger.info(f"Predicting values using model '{self.model_resource.name()}' and instance '{model_key}'")

        x_hat = self.model_resource.predict(mod, self.parse_data(x, orient=orient), **kwargs)

        if as_array:
            x_resp = x_hat.values.tolist()
        else:
            x_resp = x_hat.to_json(orient=orient)

        resp = {
            'data': x_resp,
            'orient': orient,
        }

        return resp, HTTP_200

    @custom_error
    def _get(self, model_key):
        status = self.model_resource.check_status(model_key)

        return {'status': status}, HTTP_200

    @custom_error
    def _patch(self, model_key: str, x: str, orient: str, y: str = None):
        status = self.model_resource.check_status(model_key)

        if status != ModelStatus.Done:
            return {'status': status}, HTTP_200

        model = self.model_resource.load(model_key)
        x = self.parse_data(x, orient=orient)

        kwargs = dict()
        if y is not None:
            kwargs['y'] = self.parse_data(y, orient=orient)

        # ===== Let it persist run first ===== #
        session = self.model_resource.initialize_training(model_key)

        # ===== Start background task ===== #
        self.queue_meth(model_key, self.model_resource.do_update, model, x, session)

        return {'status': self.model_resource.check_status(model_key)}, HTTP_200
