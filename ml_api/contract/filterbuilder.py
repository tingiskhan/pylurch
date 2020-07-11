from operator import eq, le, lt, ge, gt, ne, and_, or_
from sqlalchemy import bindparam, DateTime, Date, Enum
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from dateparser import parse
from typing import Dict, Union


MAPPING = {
    'eq': eq,
    'le': le,
    'lt': lt,
    'ge': ge,
    'gt': gt,
    'ne': ne,
    'and_': and_,
    'or_': or_
}

INVERSE_MAP = {v: k for k, v in MAPPING.items()}


DESERIALIZERS = {
    DateTime.__name__: lambda u: parse(u),
    Date.__name__: lambda u: parse(u).date()
}


SERIALIZERS = {
    DateTime.__name__: str,
    Date.__name__: str,
    Enum.__name__: lambda u: u.value
}


# TODO: Enum support is lacking now as enum value must be strictly equal to enum name...
class FilterBuilder(object):
    def __init__(self, obj):
        """
        Class for building filters from BinaryExpression or from json.
        :param obj: The object
        :type obj: type
        """

        self._obj = obj

    def to_json(self, be) -> Union[Dict[str, str], Dict[str, Dict[str, str]]]:
        """
        Constructs a JSON representation of a filter.
        :param be: The binary expression
        :type be: sqlalchemy.BinaryExpression
        :return: A dictionary of strings
        """

        if isinstance(be, BooleanClauseList):
            res = {
                'left': self.to_json(be.clauses[0]),
                'right': self.to_json(be.clauses[1]),
                'op': INVERSE_MAP[be.operator]
            }

            return res

        attr = getattr(self._obj, be.left.name)
        key = attr.type.__class__.__name__

        res = {
            'left': be.left.name,
            'right': be.right.value if key not in SERIALIZERS else SERIALIZERS[key](be.right.value),
            'op': INVERSE_MAP[be.operator]
        }

        return res

    def from_json(self, json) -> Union[BinaryExpression, BooleanClauseList]:
        """
        Constructs a filter from a JSON representation
        :param json: The JSON
        :type json: dict[str, str]|list[dict[str, str]]
        :return: A BinaryExpression object
        """

        left = json['left']
        right = json['right']

        left_is_dict = isinstance(left, dict)
        if left_is_dict:
            left = self.from_json(left)

        right_is_dict = isinstance(right, dict)
        if right_is_dict:
            right = self.from_json(right)

        if left_is_dict and right_is_dict:
            return MAPPING[json['op']](left, right)

        attr = getattr(self._obj, left)
        key = attr.type.__class__.__name__

        if key in DESERIALIZERS:
            value = DESERIALIZERS[key](right)
        else:
            value = json['right']

        return BinaryExpression(attr, bindparam(None, value, type_=attr.type), MAPPING[json['op']])