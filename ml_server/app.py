from flask import Flask
from flask_restful import Api
from flask_executor import Executor
from flask_bcrypt import Bcrypt
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from flask_httpauth import HTTPTokenAuth, HTTPBasicAuth
import logging


# ===== Setup service ===== #
app = Flask(__name__)
api = Api(app)
bcrypt = Bcrypt(app)
executor = Executor(app)

# ===== Logging ===== #
app.logger.setLevel(logging.DEBUG)

# ===== Auth ====== #
auth_token = HTTPTokenAuth(scheme='Token')
auth_basic = HTTPBasicAuth()
admin_auth = HTTPBasicAuth()

# ===== Read config ====== #
if 'FWS_CONFIG' in os.environ:
    app.config.from_envvar('FWS_CONFIG')
    app.logger.info('Successfully loaded config from environment variable')

sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

app.config.setdefault('SECRET_KEY', os.urandom(24))
app.config.setdefault('DB_ADDRESS', os.environ.get('DB_CONNSTRING') or sqlite)
app.config.setdefault('TOKEN_EXPIRATION', 12 * 10 * 60)
app.config.setdefault('EXECUTOR_PROPAGATE_EXCEPTIONS', True)
app.config.setdefault('PRODUCTION', os.environ.get('PRODUCTION') or False)
app.config.setdefault('EXTENSION', 'pkl')
app.config.setdefault('EXTERNAL_AUTH', os.environ.get('EXTERNAL_AUTH') or True)

# ===== Define engines and session ====== #
ENGINE = create_engine(app.config.get('DB_ADDRESS'))
SESSION = sessionmaker(bind=ENGINE)

# ===== Define tables ====== #
from .db.models import Base, User

Base.metadata.create_all(ENGINE)

app.logger.info('Successfully connected to database and created relevant tables')

# ===== Create a root user ===== #
session = SESSION()

if not session.query(User).filter(User.username == 'admin').one_or_none():
    admin = User('admin', os.environ.get('ML_ADMIN_PW', 'this-is-not-a-good-password'), admin=True)
    session.add(admin)
    session.commit()

    app.logger.info('Successfully created an admin user')

# ===== Define model manager ====== #
from .model_managers import SQLModelManager

MODEL_MANAGER = SQLModelManager(app.logger, SESSION)
MODEL_MANAGER.close_all_running()
app.logger.info(f'Application is configured as: {"production" if app.config["PRODUCTION"] else "debug"}')

# ===== Authorization logic ===== #
@auth_basic.verify_password
def verify(username, password):
    session = SESSION()

    user = session.query(User).filter(User.username == username).one()
    verified = user.verify_password(password)

    session.close()

    return verified


@auth_token.verify_token
def verify_token(token):
    return User.verify_auth_token(token)


# TODO: Duplication
@admin_auth.verify_password
def verify(username, password):
    session = SESSION()

    user = session.query(User).filter(User.username == username).one()
    verified = user.verify_password(password) and user.admin

    session.close()

    return verified

# ===== Define views ===== #
from .views.example import HelloWorld
from .views.admin import AdminView
from .views.regression import LinearRegressionView, LogisticRegressionView

api.add_resource(HelloWorld, '/hello-world')
api.add_resource(AdminView, '/admin')
api.add_resource(LinearRegressionView, '/linreg')
api.add_resource(LogisticRegressionView, '/logreg')

if not app.config['EXTERNAL_AUTH']:
    from .views.registration import LoginView, RegistrationView

    api.add_resource(LoginView, '/login')
    api.add_resource(RegistrationView, '/register')

    app.logger.info('Registering login views')
else:
    app.logger.info('Using external authentication')

app.logger.info('Finished setting up application')