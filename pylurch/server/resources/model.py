import pandas as pd
from pylurch.utils import custom_error
from pylurch.contract.enums import Status
from typing import Dict
from ..inference import InferenceModel
import pylurch.contract.schemas as sc
from falcon.status_codes import HTTP_200, HTTP_400
from ..tasking.wrapper import BaseWrapper


class ModelResource(object):
    def __init__(self, model_resource: InferenceModel, manager: BaseWrapper):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param manager: The task manager
        """

        self.model_resource = model_resource
        self.manager = manager

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

    @custom_error
    def _put(self, x: str, orient: str, name: str, y: str = None, modkwargs: Dict[str, object] = None,
             algkwargs: Dict[str, object] = None):
        # ===== Get data ===== #
        x = self.model_resource.parse_data(x, orient=orient)

        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()

        if y is not None:
            akws['y'] = self.model_resource.parse_data(y, orient=orient)

        # ===== Check if model exists ===== #
        model = self.model_resource.load(name)

        if model is not None:
            self.logger.info(f"Instance '{name}' of '{self.model_resource.name()}' already exists")
            return {"status": Status.Done, "session_name": name}, HTTP_200

        # ===== Define model ===== #
        model = self.model_resource.make_model(**modkwargs)

        # ===== Start background task ===== #
        key = self.manager.enqueue(self.model_resource.do_run, model, x, name=name, **akws)

        return {"task_id": key, "status": self.manager.check_status(key), "session_name": name}, HTTP_200

    @custom_error
    def _post(self, name: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, object]):
        model = self.model_resource.load(name)

        if model is None:
            self.logger.info(f"No model of '{self.model_resource.name()}' and instance '{name}' exists")
            return {"task_id": None, "status": Status.Unknown}, HTTP_400

        self.logger.info(f"Predicting values using model '{self.model_resource.name()}' and instance '{name}'")
        key = self.manager.enqueue(self.model_resource.do_predict, model, x, orient, as_array=as_array, **kwargs)

        return {"task_id": key, "status": self.manager.check_status(key)}, HTTP_200

    @custom_error
    def _get(self, task_id):
        status = self.manager.check_status(task_id)

        resp = {"status": status}
        if status != Status.Done:
            return resp, HTTP_200

        result = self.manager.get_result(task_id)

        self.logger.info(f"Result corresponds to: {result}")

        if result is None:
            return resp, HTTP_200

        resp.update(**result)

        return resp, HTTP_200

    # TODO: Needs fixing
    @custom_error
    def _patch(self, name: str, x: str, orient: str, old_name: str, y: str = None):
        model = self.model_resource.load(old_name)

        if model is None:
            self.logger.info(f"No model of '{self.model_resource.name()}' and instance '{name}' exists")
            return {"status": Status.Unknown}, HTTP_400

        x = self.model_resource.parse_data(x, orient=orient)

        kwargs = dict()
        if y is not None:
            kwargs["y"] = self.model_resource.parse_data(y, orient=orient)

        # ===== Let it persist run first ===== #
        key = self.manager.enqueue(self.model_resource.do_update, model, x, old_name=old_name, name=name, **kwargs)

        return {"status": self.manager.check_status(name), "task_id": key, "session_name": name}, HTTP_200
