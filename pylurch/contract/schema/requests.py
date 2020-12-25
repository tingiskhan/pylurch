from marshmallow import Schema, fields as f


class FitParser(Schema):
    x = f.String(required=True)
    orient = f.String(required=True)
    y = f.String(required=False)
    name = f.String(required=True)


class PutRequest(FitParser):
    model_kwargs = f.Dict(required=False, missing=dict())
    alg_kwargs = f.Dict(required=False, missing=dict())
    labels = f.List(f.String, required=False, missing=list(), allow_none=True)


class GetRequest(Schema):
    task_id = f.String(required=True)


class PatchRequest(FitParser):
    old_name = f.String(required=True)
    labels = f.List(f.String, required=False, missing=list())


class PostRequest(FitParser):
    as_array = f.Boolean(required=False, missing=False)
    kwargs = f.Dict(required=False, missing=dict())
