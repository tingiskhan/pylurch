import pyparsing as pp
from operator import eq, le, lt, ge, gt, ne, and_, or_
from sqlalchemy import bindparam, DateTime, Date, Enum, String
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from dateparser import parse
from typing import Dict, Union, Type, List
from .database import BaseMixin


MAPPING = {
    '==': eq,
    '<=': le,
    '<': lt,
    '>=': ge,
    '>': gt,
    '!=': ne,
    '&&': and_,
    '||': or_
}

INVERSE_MAP = {v: k for k, v in MAPPING.items()}


DESERIALIZERS = {
    DateTime.__name__: lambda u: parse(u),
    Date.__name__: lambda u: parse(u).date()
}


SERIALIZERS = {
    DateTime.__name__: lambda u: u.isoformat(),
    Date.__name__: lambda u: u.isoformat(),
    Enum.__name__: lambda u: f"'{u.value}'",
    String.__name__: lambda u: f"'{u}'"
}


# Adapted from: https://stackoverflow.com/questions/11133339/parsing-a-complex-logical-expression-in-pyparsing-in-a-binary-tree-fashion
class TreeParser(object):
    def __init__(self, obj: Type[BaseMixin]):
        """
        Class for building filters from BinaryExpression or from json.
        :param obj: The object
        """

        self._obj = obj
        self._expr = self._build_parser()

    @staticmethod
    def _build_parser():
        operator = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
        comparison_term = (
                pp.pyparsing_common.iso8601_datetime.copy() |
                pp.pyparsing_common.iso8601_date.copy() |
                pp.pyparsing_common.number.copy() |
                pp.QuotedString("'")
        )
        condition = pp.Group(pp.pyparsing_common.identifier + operator + comparison_term)

        expr = pp.operatorPrecedence(
            condition,
            [
                ("&&", 2, pp.opAssoc.LEFT),
                ("||", 2, pp.opAssoc.LEFT),
            ]
        )

        return expr

    def _recursion(self, expr: pp.ParseResults):
        length = len(expr)
        if length == 3 and not isinstance(expr[0], pp.ParseResults):
            attr = getattr(self._obj, expr[0])
            op = expr[1]
            key = attr.type.__class__.__name__
            value = expr[2] if key not in DESERIALIZERS else DESERIALIZERS[key](expr[2])

            if isinstance(attr.type, Enum):
                value = attr.type.python_type[value]

            return BinaryExpression(attr, bindparam(None, value, type_=attr.type), MAPPING[op])

        mid = length // 3 + length % 3

        left = self._recursion(expr[:mid] if length > 3 else expr[0])
        right = self._recursion(expr[-mid:] if length > 3 else expr[-1])
        operator = expr[mid]

        return MAPPING[operator](left, right)

    def to_string(self, expression: Union[BooleanClauseList, BinaryExpression]):
        if isinstance(expression, BooleanClauseList):
            left = self.to_string(expression.clauses[0])
            right = self.to_string(expression.clauses[1])

            return f"({left} {INVERSE_MAP[expression.operator]} {right})"

        attr = getattr(self._obj, expression.left.name)
        key = attr.type.__class__.__name__

        left = expression.left.name
        right = expression.right.value if key not in SERIALIZERS else SERIALIZERS[key](expression.right.value)

        return f"({left}{INVERSE_MAP[expression.operator]}{right})"

    def from_string(self, expression: str):
        root = self._expr.parseString(expression)[0]

        return self._recursion(root)
