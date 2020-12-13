from starlette.applications import Starlette
from pylurch.server.resources import ModelResource
from pylurch.contract.interface import SessionInterface
from pyalfred.server.utils import make_base_logger
from pylurch.server.tasking.runners import RQRunner
from redis import Redis
import os


def init_app():
    api = Starlette()

    # ===== Add models ===== #
    from .models import LinearRegressionModel, LogisticRegressionModel

    intf = SessionInterface(os.environ.get("DATABASE_URI"))
    manager = RQRunner(Redis(host=os.environ.get("REDIS_HOST"), port=os.environ.get("REDIS_PORT")), intf)

    api.add_route("/linreg", ModelResource.make_endpoint(LinearRegressionModel(), manager, intf))
    api.add_route("/logreg", ModelResource.make_endpoint(LogisticRegressionModel(), manager, intf))

    logger = make_base_logger(__name__)
    logger.info("Successfully registered all views")

    return api
