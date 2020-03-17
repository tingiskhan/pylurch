from enum import Enum


class ModelStatus(Enum):
    Running = 'RUNNING'
    Done = 'FINISHED'
    Failed = 'FAILED'
    Cancelled = 'CANCELLED'


class SerializerBackend(Enum):
    Dill = 'dill'
    ONNX = 'onnx'