from flask_restful import Resource
from ..app import admin_auth, ac
from ..utils import custom_login
from flask import request


class AdminView(Resource):
    @custom_login(admin_auth)
    def delete(self):
        request.environ.get('werkzeug.server.shutdown')()

        ac.app.logger.info('Shutting down service, might take a while')

        return {'message': 'Killing service, might take a while as background workers need to finish'}
