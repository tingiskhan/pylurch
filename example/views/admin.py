from flask_restful import Resource
from ..app import admin_auth, app
from flask import request


class AdminView(Resource):
    @admin_auth.login_required
    def delete(self):
        request.environ.get('werkzeug.server.shutdown')()

        app.logger.info('Shutting down service, might take a while')

        return {'message': 'Killing service, might take a while as background workers need to finish'}
