from flask_restful import Resource
from ..app import auth_basic, app
from ..utils import custom_login
from flask import request


class AdminView(Resource):
    @custom_login(auth_basic)
    def delete(self):
        request.environ.get('werkzeug.server.shutdown')()

        app.logger.info('Shutting down service, might take a while')

        return {'message': 'Killing service, might take a while as background workers need to finish'}
