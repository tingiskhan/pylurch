class Mixin(object):
    @classmethod
    def __contains__(cls, item):
        attrs = vars(cls)

        return item in attrs.values()

    @classmethod
    def __str__(cls):
        return str(list(v for k, v in vars(cls).items() if not (v or '__').startswith('__')))


class ModelStatus(Mixin):
    Running = 'RUNNING'
    Done = 'DONE'
    Failed = 'FAILED'
    Cancelled = 'CANCELLED'


class SerializerBackend(Mixin):
    Dill = 'dill'
    ONNX = 'onnx'