from enum import Enum


class ModelStatus(Enum):
    Running = 'RUNNING'
    Done = 'FINISHED'
    Failed = 'FAILED'
    Cancelled = 'CANCELLED'


class SerializerBackend(Enum):
    Dill = 'dill'
    ONNX = 'onnx'


EXECUTOR_MAP = {
    'RUNNING': ModelStatus.Running,
    'FINISHED': ModelStatus.Done,
    'FAILED': ModelStatus.Failed
}

