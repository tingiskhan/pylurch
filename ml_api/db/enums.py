from enum import Enum


class ModelStatus(Enum):
    Running = 'RUNNING'
    Done = 'FINISHED'
    Failed = 'FAILED'
    Cancelled = 'CANCELLED'


class SerializerBackend(Enum):
    Dill = 'dill'
    ONNX = 'onnx'
    Custom = 'custom'


EXECUTOR_MAP = {
    'RUNNING': ModelStatus.Running,
    'FINISHED': ModelStatus.Done,
    'FAILED': ModelStatus.Failed
}

