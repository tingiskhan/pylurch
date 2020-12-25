from .base import LoadableClientContext


class ClientPredictionContext(LoadableClientContext):
    def on_exit(self):
        return
