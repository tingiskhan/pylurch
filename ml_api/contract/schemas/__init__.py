from .argument import PatchParser, PutParser, GetParser, PostParser
from .responses import PatchResponse, PutResponse, GetResponse, PostResponse
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema


class BaseSchema(SQLAlchemyAutoSchema):
    @classmethod
    def endpoint(cls):
        if not cls.__name__.endswith('Schema'):
            raise NotImplementedError('All schemas must end with `Schema`!')

        return cls.__name__.lower().replace('schema', '')


from .model import TrainingSessionSchema, ModelSchema, MetaDataSchema