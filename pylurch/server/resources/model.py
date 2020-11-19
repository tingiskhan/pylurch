from pylurch.utils import custom_error
from pylurch.contract.enums import Status
from typing import Dict, Any, List
from ..inference import InferenceModel, ModelWrapper
import pylurch.contract.schemas as sc
from falcon.status_codes import HTTP_200, HTTP_400
from ..tasking.wrapper import BaseWrapper
from pylurch.contract.interfaces import DatabaseInterface


class ModelResource(object):
    def __init__(self, model_resource: InferenceModel, manager: BaseWrapper, intf: DatabaseInterface):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param manager: The task manager
        """

        self.wrap = ModelWrapper(model_resource, intf)
        self.manager = manager

    @property
    def logger(self):
        return self.wrap.logger

    def _apply_and_parse(self, meth, request, res, parser, resp):
        if meth == self._get:
            temp, status = meth(**parser.load(request.params))
        else:
            temp, status = meth(**parser.load(request.media))

        res.media = resp.dump(temp)
        res.status = status

        return res

    def on_get(self, req, res):
        return self._apply_and_parse(self._get, req, res, sc.GetRequest(), sc.GetResponse())

    def on_put(self, req, res):
        return self._apply_and_parse(self._put, req, res, sc.PutRequest(), sc.PutResponse())

    def on_post(self, req, res):
        return self._apply_and_parse(self._post, req, res, sc.PostRequest(), sc.PostResponse())

    def on_patch(self, req, res):
        return self._apply_and_parse(self._patch, req, res, sc.PatchRequest(), sc.PatchResponse())

    @custom_error
    def _put(
        self,
        x: str,
        orient: str,
        name: str,
        y: str = None,
        modkwargs: Dict[str, Any] = None,
        algkwargs: Dict[str, Any] = None,
        labels: List[str] = None,
    ):
        # ===== Get data ===== #
        x = self.wrap.model.parse_x(x, orient=orient)

        modkwargs = modkwargs or dict()
        akws = algkwargs or dict()
        labels = labels or list()

        if y is not None:
            akws["y"] = self.wrap.model.parse_y(y, orient=orient)

        # ===== Start background task ===== #
        key = self.manager.enqueue(self.wrap.do_run, modkwargs, x, name=name, labels=labels, **akws)

        return {"task_id": key, "status": self.manager.check_status(key), "name": name}, HTTP_200

    @custom_error
    def _post(self, name: str, x: str, orient: str, as_array: bool, kwargs: Dict[str, Any]):
        exists = self.wrap.session_exists(name)

        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{name}' exists")
            return {"task_id": None, "status": Status.Unknown}, HTTP_400

        self.logger.info(f"Predicting values using model '{self.wrap.model.name()}' and instance '{name}'")
        key = self.manager.enqueue(self.wrap.do_predict, name, x, orient, as_array=as_array, **kwargs)

        return {"task_id": key, "status": self.manager.check_status(key)}, HTTP_200

    @custom_error
    def _get(self, task_id):
        status = self.manager.check_status(task_id)

        resp = {"status": status}
        if status != Status.Done:
            return resp, HTTP_200

        result = self.manager.get_result(task_id)
        if result is None:
            return resp, HTTP_200

        self.logger.info(f"Got result for task id: {task_id}")

        resp.update(**result)

        return resp, HTTP_200

    @custom_error
    def _patch(self, name: str, x: str, orient: str, old_name: str, y: str = None, labels: List[str] = None):
        exists = self.wrap.session_exists(old_name)

        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{name}' exists")
            return {"status": Status.Unknown}, HTTP_400

        x = self.wrap.model.parse_data(x, orient=orient)

        kwargs = dict()
        if y is not None:
            kwargs["y"] = self.wrap.model.parse_data(y, orient=orient)

        # ===== Let it persist run first ===== #
        key = self.manager.enqueue(self.wrap.do_update, old_name, x, name=name, labels=labels, **kwargs)

        return {"status": self.manager.check_status(name), "task_id": key, "name": name}, HTTP_200
