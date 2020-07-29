from ...database import BaseMixin, SERIALIZATION_IGNORE
from typing import List, Callable
from copy import copy
from requests import post, put, delete, patch
from ...filterbuilder import FilterBuilder
from ...utils import chunk, Constants
from ...schemas import BaseSchema
from typing import Type
from ..base import BaseInterface


class DatabaseInterface(BaseInterface):
    def __init__(self, base, base_schema=BaseSchema):
        """
        An interface for defining and creating.
        """
        super().__init__(base, '')
        self._schema = None
        self._base_schema = base_schema

    def make_interface(self, obj: Type[BaseMixin]):
        """
        Makes an interface to the specified schema
        :param obj: The object to create for
        :return: DataBaseInterface
        :rtype: DatabaseInterface
        """

        cp = copy(self)
        schema = self._base_schema.get_schema(obj)
        cp._ep = schema.endpoint()
        cp._schema = schema

        return cp

    def _deserialize(self, dct, **kwargs):
        res = self._schema(**kwargs).load(dct)

        if isinstance(res, dict):
            return self._schema.Meta.model(**res)

        return tuple(self._schema.Meta.model(**it) for it in res)

    def _serialize(self, obj, **kwargs):
        return self._schema(**kwargs).dump(obj)

    def create(self, obj: BaseMixin or List[BaseMixin]) -> BaseMixin or List[BaseMixin]:
        """
        Create an object of type specified by Meta object in `schema`.
        :param obj: The object, or objects
        :return: An object of type specified by Meta object in `schema`
        """

        if not isinstance(obj, (list, tuple)):
            obj = [obj]

        res = list()
        for c in chunk(obj, Constants.ChunkSize.value):
            dump = self._serialize(c, load_only=SERIALIZATION_IGNORE, many=True)
            req = self._exec_req(put, json=dump)
            res.extend(self._deserialize(req, many=True))

        if len(obj) < 2:
            return res[0]

        return res

    def get(self, f: Callable[[BaseMixin], bool] = None, one: bool = False) -> BaseMixin or List[BaseMixin]:
        """
        Get an object of type specified by Meta object in `schema`.
        :param f: A callable with 1 parameter for constructing a BinaryExpression
        :param one: Whether to get only one
        :return: The object of type specified by Meta object in `schema`, or all
        """

        json = None
        if f:
            fb = FilterBuilder(self._schema.Meta.model)
            json = fb.to_json(f(self._schema.Meta.model))

        req = self._exec_req(post, json=json)
        res = self._deserialize(req, many=True)

        if not one:
            return res

        if len(res) > 1:
            raise ValueError('More than 1 elements exist!')

        if len(res) == 1:
            return res[0]

        return None

    def delete(self, objs: BaseMixin or List[BaseMixin]) -> int:
        """
        Deletes an object with `id_` of type specified by Meta object in `schema`.
        :param objs: The objects to delete
        :return: The number of affected items
        """

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        deleted = 0
        for obj in objs:
            req = self._exec_req(delete, params={'id': obj.id})
            deleted += req['deleted']

        return deleted

    def update(self, objs: BaseMixin or List[BaseMixin]) -> List[BaseMixin]:
        """
        Updates an object with the new values.
        :param objs: The object(s) to update with new values
        :return: The update object
        """

        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        dump = self._schema(many=True).dump(objs)
        req = self._exec_req(patch, json=dump)

        return self._deserialize(req, many=True)