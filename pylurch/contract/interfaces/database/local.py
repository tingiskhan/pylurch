from .remote import DatabaseInterface
from ...utils import chunk, Constants
from logging import Logger
from sqlalchemy.orm import Session, scoped_session
from typing import Union
from ....utils import make_base_logger


class LocalDatabaseInterface(DatabaseInterface):
    def __init__(self, session: Union[Session, scoped_session], logger: Logger = None, **kwargs):
        """
        Interface for communicating with a database directly. Useful if you decide to expose the data models and
        ML models on the same server instance.
        :param session: Session manager
        """

        super().__init__('', **kwargs)
        self._session = session
        self.logger = logger or make_base_logger(self.__class__.__name__)

    def session_factory(self):
        return self._session()

    def create(self, objs):
        session = self.session_factory()

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        for c in chunk(objs, Constants.ChunkSize.value):
            session.add_all(c)
            session.flush()
        try:
            session.commit()
            self.logger.info(f'Successfully created {len(objs):n} objects, now trying to serialize')
        except Exception as e:
            self.logger.exception(e)
            session.rollback()

        res = self._serialize(objs, many=True)
        session.close()
        res = self._deserialize(res, many=True)

        if len(res) < 2:
            return res[0]

        return res

    def update(self, objs):
        session = self.session_factory()

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        self.logger.info(f'Now trying to update {len(objs):n} objects')

        for obj in objs:
            session.merge(obj)

        try:
            session.commit()
            self.logger.info(f'Successfully update {len(objs):n} objects, now trying to serialize')
        except Exception as e:
            self.logger.exception(e)
            session.rollback()

        res = self._serialize(objs, many=True)
        session.close()
        return self._deserialize(res, many=True)

    def delete(self, objs):
        session = self.session_factory()

        if not isinstance(objs, (tuple, list)):
            objs = [objs]

        nums = 0
        try:
            model = self._schema.Meta.model
            nums = session.query(model).filter(model.id.in_(o.id for o in objs)).delete('fetch')
            self.logger.info(f'Now trying to delete {nums:n} objects')
            session.commit()

            self.logger.info(f'Successfully deleted {nums:n} objects')
        except Exception as e:
            self.logger.exception(e)
            session.rollback()

        session.close()
        return nums

    def get(self, f=None, one=False):
        session = self.session_factory()
        query = session.query(self._schema.Meta.model)

        if f is not None:
            query = query.filter(f(self._schema.Meta.model))

        res = self._serialize(query.all(), many=True)
        session.close()
        res = self._deserialize(res, many=True)

        if not one:
            return res

        if len(res) > 1:
            raise ValueError('More than 1 elements exist!')

        if len(res) == 1:
            return res[0]

        return None