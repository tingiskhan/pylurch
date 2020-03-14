class BaseLogHelper(object):
    def __init__(self, app):
        """
        Base helper for defining logging.
        :param app: The application
        :type app: flask.Flask
        """

        self._app = app

    def add_handler(self):
        """
        Method to override for adding handlers.
        :return: Self
        :rtype: BaseLogHelper
        """

        raise NotImplementedError()