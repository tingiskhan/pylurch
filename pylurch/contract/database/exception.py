from . import BaseMixin
from sqlalchemy import Column, String


class ExceptionTemplate(BaseMixin):
    type_ = Column(String(), nullable=False)
    message = Column(String(), nullable=False)
