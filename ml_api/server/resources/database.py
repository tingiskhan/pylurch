from ml_api.contract.schemas import BaseSchema
from ml_api.contract.filterbuilder import FilterBuilder
from typing import Union
from ml_api.contract.utils import chunk, Constants
from sqlalchemy.orm import scoped_session, sessionmaker
from falcon.status_codes import HTTP_500
from logging import Logger
from ...utils import make_base_logger


class DatabaseResource(object):
    def __init__(self, schema: BaseSchema, session_factory: Union[scoped_session, sessionmaker], logger: Logger = None):
        """
        Implements a base resources for exposing database models.
        :param schema: The schema to use, must be marshmallow.Schema
        :param session_factory: The sqlalchemy scoped_session object to use
        :param logger: The logger to use
        """

        self.schema = schema
        self.session_factory = session_factory
        self.logger = logger or make_base_logger(schema.endpoint())

    @property
    def model(self):
        return self.schema.Meta.model

    def deserialize(self, json, **kwargs):
        schema = self.schema(**kwargs)
        conf = schema.load(json)

        if isinstance(conf, dict):
            return self.model(**conf)

        return list(self.model(**it) for it in conf)

    def serialize(self, objs, **kwargs):
        schema = self.schema(**kwargs)

        return schema.dump(objs)

    def on_post(self, req, res):
        session = self.session_factory()
        query = session.query(self.model)

        if req.media:
            fb = FilterBuilder(self.model)
            be = fb.from_json(req.media)
            query = query.filter(be)

        res.media = self.serialize(query.all(), many=True)
        session.close()

        return res

    def on_put(self, req, res):
        objs = self.deserialize(req.media, many=True)
        self.logger.info(f'Now trying to create {len(objs):n} objects')
        session = self.session_factory()

        for c in chunk(objs, Constants.ChunkSize.value):
            session.add_all(c)
            session.flush()

        try:
            session.commit()
            self.logger.info(f'Successfully created {len(objs):n} objects, now trying to serialize')
            res.media = self.serialize(objs, many=True)
        except Exception as e:
            self.logger.exception(e)
            res.media = {'message': 'Failed comitting to database!'}
            res.status = HTTP_500
            session.rollback()

        session.close()

        return res

    def on_delete(self, req, res):
        session = self.session_factory()

        try:
            nums = session.query(self.model).filter(self.model.id == req.params['id']).delete('fetch')
            self.logger.info(f'Now trying to delete {nums:n} objects')
            session.commit()

            self.logger.info(f'Successfully deleted {nums:n} objects')
            res.media = {'deleted': nums}
        except Exception as e:
            self.logger.exception(e)
            session.rollback()
            res.media = {'message': 'Failed deleting in database!'}
            res.status = HTTP_500

        session.close()
        return res

    def on_patch(self, req, res):
        objs = self.deserialize(req.media, many=True)
        session = self.session_factory()
        self.logger.info(f'Now trying to update {len(objs):n} objects')

        for obj in objs:
            session.merge(obj)

        try:
            session.commit()
            self.logger.info(f'Successfully update {len(objs):n} objects, now trying to serialize')
            res.media = self.serialize(objs, many=True)
        except Exception as e:
            self.logger.exception(e)
            session.rollback()
            res.media = {'message': 'Failed commiting to database!'}
            res.status = HTTP_500

        session.close()
        return res