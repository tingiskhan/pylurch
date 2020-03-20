from flask_restful import Resource
from ..app import ac


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        ac.app.logger.warning('This is just a test')

    def put(self):
        raise NotImplementedError('This is also a test')