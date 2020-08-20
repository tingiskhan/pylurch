from marshmallow import Schema, fields as f


class FitParser(Schema):
    x = f.String(required=True)
    orient = f.String(required=True)
    y = f.String(required=False)
    name = f.String(required=True)


class PutParser(FitParser):
    modkwargs = f.Dict(required=False, missing=dict())
    algkwargs = f.Dict(required=False, missing=dict())


class GetParser(Schema):
    task_id = f.String(required=True)


class PatchParser(FitParser):
    old_name = f.String(required=True)


class PostParser(FitParser):
    as_array = f.Boolean(required=False, missing=False)
    kwargs = f.Dict(required=False, missing=dict())





