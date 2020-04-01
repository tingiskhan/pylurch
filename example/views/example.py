from flask_restful import Resource
from ..app import app


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        app.logger.warning('This is just a test')

    def put(self):
        raise NotImplementedError('This is also a test')