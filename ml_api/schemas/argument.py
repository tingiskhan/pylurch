from marshmallow import Schema, fields as f


class FitParser(Schema):
    x = f.String(required=True)
    orient = f.String(required=True)
    y = f.String(required=False)


class PutParser(FitParser):
    name = f.String(required=True)
    modkwargs = f.Dict(required=False, missing=dict())
    algkwargs = f.Dict(required=False, missing=dict())
    retrain = f.Boolean(required=False, missing=False)


class GetParser(Schema):
    model_key = f.String(required=True)


class PatchParser(FitParser, GetParser):
    model_key = f.String(required=True)


class PostParser(PatchParser):
    as_array = f.Boolean(required=False, missing=False)
    kwargs = f.Dict(required=False, missing=dict())





