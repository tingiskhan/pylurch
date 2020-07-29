from .argument import PatchParser, PutParser, GetParser, PostParser
from .responses import PatchResponse, PutResponse, GetResponse, PostResponse
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema


class BaseSchema(SQLAlchemyAutoSchema):
    @classmethod
    def endpoint(cls):
        if not cls.__name__.endswith('Schema'):
            raise NotImplementedError('All schemas must end with `Schema`!')

        return cls.__name__.lower().replace('schema', '')

    @classmethod
    def get_schemas(cls):
        subs = cls.__subclasses__()

        for s in subs:
            subs.extend(s.get_schemas())

        return subs

    @classmethod
    def get_schema(cls, obj):
        return next(s for s in cls.get_schemas() if s.Meta.model == obj)


from .model import TrainingSessionSchema, ModelSchema, MetaDataSchema, UpdatedSessionSchema
from .task import TaskSchema, TaskMetaSchema