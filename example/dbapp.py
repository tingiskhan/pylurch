from falcon import API as Api
from pylurch.server.resources import DatabaseResource
from pylurch.contract.database import Base
from pylurch.contract.schemas import DatabaseSchema
from pylurch.utils import make_base_logger
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import os


def init_app():
    # ===== Database related ===== #
    engine = create_engine(
        os.environ.get('SQLALCHEMY_DATABASE_URI', "sqlite:///debug-database.db?check_same_thread=false"),
        **os.environ.get('SQLALCHEMY_ENGINE_OPTIONS', {'pool_pre_ping': True})
    )

    Session = scoped_session(sessionmaker(bind=engine))

    # ===== Initialize everything ===== #
    Base.metadata.create_all(bind=engine)

    api = Api()

    for base in DatabaseSchema.get_subclasses(Base):
        s = DatabaseSchema.generate_schema(base)
        api.add_route(f"/{s.endpoint()}", DatabaseResource(s, Session))

    logger = make_base_logger(__name__)
    logger.info("Successfully registered all views")

    return api