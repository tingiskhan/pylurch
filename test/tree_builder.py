import unittest
from pylurch.contract.treebuilder import TreeParser
from pylurch.contract.database import BaseMixin
from sqlalchemy.ext.declarative import declarative_base, declared_attr
import platform
from sqlalchemy import Column, String, DateTime, Integer
from datetime import datetime


Base = declarative_base()


class Dummy(BaseMixin, Base):
    x = Column(String(255), nullable=False, unique=True)


# TODO: Implement proper tests
class MyTestCase(unittest.TestCase):
    def test_treeBuilderFromString(self):
        pass

    def test_treeBuilderToString(self):
        pass


if __name__ == '__main__':
    unittest.main()
