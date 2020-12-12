from falcon.errors import HTTPBadRequest
from pylurch.contract.enums import Status
from typing import Dict, Any, List
from falcon.status_codes import HTTP_200, HTTP_400, HTTP_500
import pylurch.contract.schema as sc
from pylurch.contract.interface import SessionInterface
from ..tasking.runners.base import BaseRunner
from ..inference import InferenceModel, ModelWrapper


class ModelResource(object):
    def __init__(self, model_resource: InferenceModel, manager: BaseRunner, intf: SessionInterface, **kwargs):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param manager: The task manager
        """

        self.wrap = ModelWrapper(model_resource, intf, **kwargs)
        self.manager = manager

    @property
    def logger(self):
        return self.wrap.logger

    def _apply_and_parse(self, meth, request, res, parser, resp):
        try:
            if meth == self.get_result:
                func_result, status = meth(**parser.load(request.params))
            else:
                func_result, status = meth(**parser.load(request.media))

        except Exception as e:
            self.logger.exception("Failed in task", e)
            if isinstance(e, HTTPBadRequest):
                raise e

            status = HTTP_500
            func_result = {"message": repr(e)}

        res.media = resp.dump(func_result)
        res.status = status

        return res

    def on_get(self, req, res):
        return self._apply_and_parse(self.get_result, req, res, sc.GetRequest(), sc.GetResponse())

    def on_put(self, req, res):
        return self._apply_and_parse(self.train, req, res, sc.PutRequest(), sc.PutResponse())

    def on_post(self, req, res):
        return self._apply_and_parse(self.predict, req, res, sc.PostRequest(), sc.PostResponse())

    def on_patch(self, req, res):
        return self._apply_and_parse(self.update, req, res, sc.PatchRequest(), sc.PatchResponse())

    def train(
        self,
        x: str,
        orient: str,
        name: str,
        y: str = None,
        modkwargs: Dict[str, Any] = None,
        algkwargs: Dict[str, Any] = None,
        labels: List[str] = None,
    ):
        x_d, y_d = self.wrap.model.parse_x_y(x, y=y, orient=orient)

        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()
        labels = labels or list()

        key = self.manager.enqueue(self.wrap.do_run, modkwargs, x_d, name=name, labels=labels, y=y_d, **akws)

        return {"task_id": key, "status": self.manager.check_status(key), "name": name}, HTTP_200

    def predict(self, name: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, Any]):
        exists = self.wrap.session_exists(name)

        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{name}' exists")
            return {"task_id": None, "status": Status.Unknown}, HTTP_400

        x_d, _ = self.wrap.model.parse_x_y(x, y=None, orient=orient)
        key = self.manager.enqueue(self.wrap.do_predict, name, x_d, orient, as_array=as_array, **kwargs)

        return {"task_id": key, "status": self.manager.check_status(key)}, HTTP_200

    def get_result(self, task_id):
        status = self.manager.check_status(task_id)
        resp = {"status": status}

        if status != Status.Done:
            if status == Status.Failed:
                exc = self.manager.get_exception(task_id)
                if exc is not None:
                    resp["message"] = exc.message

            return resp, HTTP_200

        result = self.manager.get_result(task_id)
        if result is None:
            return resp, HTTP_200

        self.logger.info(f"Got result for task id: {task_id}")

        resp.update(**result)

        return resp, HTTP_200

    def update(self, name: str, x: str, orient: str, old_name: str, y: str = None, labels: List[str] = None):
        exists = self.wrap.session_exists(old_name)

        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{name}' exists")
            return {"status": Status.Unknown}, HTTP_400

        x_d, y_d = self.wrap.model.parse_x_y(x, y, orient=orient)

        # ===== Let it persist run first ===== #
        key = self.manager.enqueue(self.wrap.do_update, old_name, x_d, name=name, labels=labels, y=y_d)

        return {"status": self.manager.check_status(name), "task_id": key, "name": name}, HTTP_200
