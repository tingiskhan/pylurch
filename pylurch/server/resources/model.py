from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
from starlette.endpoints import HTTPEndpoint
from pylurch.contract.enums import Status
import pylurch.contract.schema as sc
from pylurch.contract.interface import SessionInterface
from ..tasking.runners.base import BaseRunner
from pylurch.inference import InferenceModel, ModelWrapper


class ModelResource(HTTPEndpoint):
    wrap: ModelWrapper = None
    manager: BaseRunner = None

    @classmethod
    def make_endpoint(cls, model_resource: InferenceModel, manager: BaseRunner, intf: SessionInterface, **kwargs):
        """
        Base object for exposing model object.
        :param model_resource: The model resource
        :param manager: The task manager
        """

        state_dict = {"wrap": ModelWrapper(model_resource, intf, **kwargs), "manager": manager}

        return type(f"{cls.__name__}_{model_resource.name()}", (ModelResource,), state_dict)

    @property
    def logger(self):
        return self.wrap.logger

    async def get(self, req: Request):
        task_id = sc.GetRequest().load(req.query_params)["task_id"]

        status = self.manager.check_status(task_id)
        resp = {"status": status}

        dumper = sc.GetResponse()
        if status != Status.Done:
            if status == Status.Failed:
                exc = self.manager.get_exception(task_id)
                if exc is not None:
                    resp["message"] = exc.message

            return JSONResponse(dumper.dump(resp))

        result = self.manager.get_result(task_id)
        if result is None:
            return JSONResponse(dumper.dump(resp))

        self.logger.info(f"Got result for task id: {task_id}")

        resp.update(**result)

        return JSONResponse(dumper.dump(resp))

    async def put(self, req: Request):
        put_req = sc.PutRequest().load(await req.json())

        x_d, y_d = self.wrap.model.parse_x_y(put_req["x"], y=put_req["y"], orient=put_req["orient"])

        modkwargs = put_req["modkwargs"] or dict()
        akws = put_req["algkwargs"] or dict()
        labels = put_req["labels"] or list()

        name = put_req["name"]
        key = self.manager.enqueue(self.wrap.do_run, modkwargs, x_d, name=name, labels=labels, y=y_d, **akws)

        resp = sc.PutResponse().dump({"task_id": key, "status": self.manager.check_status(key), "name": name})
        return JSONResponse(resp)

    async def post(self, req: Request):
        post_req = sc.PostRequest().load(await req.json())

        name = post_req["name"]
        exists = self.wrap.session_exists(name)

        dumper = sc.PostResponse()
        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{name}' exists")
            return JSONResponse(dumper.dump({"task_id": None, "status": Status.Unknown}), HTTP_400_BAD_REQUEST)

        orient = post_req["orient"]
        x_d, _ = self.wrap.model.parse_x_y(post_req["x"], y=None, orient=orient)
        key = self.manager.enqueue(
            self.wrap.do_predict, name, x_d, orient, as_array=post_req["as_array"], **post_req["kwargs"]
        )

        resp = dumper.dump({"task_id": key, "status": self.manager.check_status(key)})
        return JSONResponse(resp)

    async def patch(self, req: Request):
        patch_req = sc.PatchRequest().load(await req.json())

        old_name = patch_req["old_name"]
        exists = self.wrap.session_exists(old_name)
        dumper = sc.PatchRequest()

        if not exists:
            self.logger.info(f"No model of '{self.wrap.model.name()}' and instance '{old_name}' exists")
            return JSONResponse(dumper.dump({"status": Status.Unknown}), HTTP_400_BAD_REQUEST)

        x_d, y_d = self.wrap.model.parse_x_y(patch_req["x"], patch_req["y"], orient=patch_req["orient"])

        name = patch_req["name"]
        key = self.manager.enqueue(self.wrap.do_update, old_name, x_d, name=name, labels=patch_req["labels"], y=y_d)

        resp = dumper.dump({"status": self.manager.check_status(name), "task_id": key, "name": name})
        return JSONResponse(resp)
