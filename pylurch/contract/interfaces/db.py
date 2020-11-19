from ..database import BaseMixin, SERIALIZATION_IGNORE
from typing import Callable
from requests import get, put, delete, patch
from ..treebuilder import TreeParser
from ..utils import chunk, Constants, serialize, deserialize
from ..schemas import DatabaseSchema
from typing import Type, TypeVar, List, Union
from .base import BaseInterface


T = TypeVar("T", bound=BaseMixin)


def decorator(f):
    def wrapper(self, objs: T or List[T], **kwargs):
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        if any(not isinstance(o, objs[0].__class__) for o in objs):
            raise ValueError()

        return f(self, objs, **kwargs)

    return wrapper


class DatabaseInterface(BaseInterface):
    def __init__(self, base):
        """
        An interface for defining and creating.
        """
        super().__init__(base, "")
        self._schema = None

    @decorator
    def create(self, objects: Union[T, List[T]]) -> Union[T, List[T]]:
        """
        Create an object of type specified by Meta object in `schema`.
        :return: An object of type specified by Meta object in `schema`
        """

        res = list()
        schema = DatabaseSchema.get_schema(type(objects[0]))
        for c in chunk(objects, Constants.InterfaceChunk.value):
            dump = serialize(c, schema, load_only=SERIALIZATION_IGNORE, many=True)
            req = self._exec_req(put, endpoint=schema.endpoint(), json=dump)
            res.extend(deserialize(req, schema, many=True))

        if len(objects) < 2:
            return res[0]

        return res

    def get(self, objtype: Type[T], f: Callable[[T], bool] = None, one=False, latest=False) -> Union[T, List[T], None]:
        """
        Get an object of type specified by Meta object in `schema`.
        :param objtype: The object type to get
        :param f: A callable with 1 parameter for constructing a BinaryExpression
        :param one: Whether to get only one
        :param latest: Whether to only returns the latest
        :return: The object of type specified by Meta object in `schema`, or all
        """

        json = None
        schema = DatabaseSchema.get_schema(objtype)
        if f:
            fb = TreeParser(schema.Meta.model)
            json = fb.to_string(f(schema.Meta.model))

        req = self._exec_req(get, endpoint=schema.endpoint(), params={"filter": json, "latest": latest})
        res = deserialize(req, schema, many=True)

        if not (one or latest):
            return res

        if len(res) > 1:
            raise ValueError("More than 1 elements exist!")

        if len(res) == 1:
            return res[0]

        return None

    @decorator
    def delete(self, objects: Union[T, List[T]]) -> int:
        """
        Deletes an object with `id_` of type specified by Meta object in `schema`.
        :param objects: The objects to delete
        :return: The number of affected items
        """

        deleted = 0
        schema = DatabaseSchema.get_schema(type(objects[0]))
        for obj in objects:
            req = self._exec_req(delete, endpoint=schema.endpoint(), params={"id": obj.id})
            deleted += req["deleted"]

        return deleted

    @decorator
    def update(self, objects: Union[T, List[T]]) -> List[T]:
        """
        Updates an object with the new values.
        :param objects: The object(s) to update with new values
        :return: The update object
        """

        res = list()
        schema = DatabaseSchema.get_schema(type(objects[0]))
        for c in chunk(objects, Constants.InterfaceChunk.value):
            dump = serialize(c, schema, many=True)
            req = self._exec_req(patch, endpoint=schema.endpoint(), json=dump)
            res.extend(deserialize(req, schema, many=True))

        return res
