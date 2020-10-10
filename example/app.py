from falcon import API as Api
from pylurch.server.resources import ModelResource
from pylurch.contract.interfaces import DatabaseInterface
from pylurch.utils import make_base_logger
from pylurch.server.tasking import RQWrapper
from redis import Redis
import os


# ===== Initialization script ===== #
def init_app():
    api = Api()

    # ===== Add models ===== #
    from .models import LinearRegressionModel, LogisticRegressionModel

    intf = DatabaseInterface(os.environ.get("DATABASE_URI"))
    manager = RQWrapper(Redis(host=os.environ.get("REDIS_HOST"), port=os.environ.get("REDIS_PORT")), intf)

    api.add_route(
        "/linreg",
        ModelResource(LinearRegressionModel(), manager, intf)
    )

    api.add_route(
        "/logreg",
        ModelResource(LogisticRegressionModel(), manager, intf)
    )

    logger = make_base_logger(__name__)
    logger.info("Successfully registered all views")

    return api