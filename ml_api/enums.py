from enum import Enum


class ModelStatus(Enum):
    Running = 'Running'
    Done = 'Done'
    Failed = 'Failed'
    Cancelled = 'Cancelled'
    Unknown = 'Unknown'


class SerializerBackend(Enum):
    Dill = 'Dill'
    ONNX = 'ONNX'
    Custom = 'Custom'
