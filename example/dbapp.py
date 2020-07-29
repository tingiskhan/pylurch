from falcon import API as Api
from pylurch.server.resources import DatabaseResource
from pylurch.contract.database import Base
from pylurch.utils import make_base_logger
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import os

# ===== Database related ===== #
sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

engine = create_engine(
    os.environ.get('SQLALCHEMY_DATABASE_URI', sqlite),
    **os.environ.get('SQLALCHEMY_ENGINE_OPTIONS', {'pool_pre_ping': True})
)

Session = scoped_session(sessionmaker(bind=engine))


# ===== Initialization script ===== #
def init_app():
    api = Api()

    # ===== Initialize everything ===== #
    Base.metadata.create_all(bind=engine)

    # ===== Add database interface ===== #
    from pylurch.contract.schemas import BaseSchema

    for o in BaseSchema.get_schemas():
        api.add_route(f'/{o.endpoint()}', DatabaseResource(o, Session))

    logger = make_base_logger(__name__)
    logger.info('Successfully registered all views')

    return api