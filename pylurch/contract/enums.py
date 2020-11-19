from enum import Enum


class Status(Enum):
    Running = "Running"
    Done = "Done"
    Failed = "Failed"
    Cancelled = "Cancelled"
    Unknown = "Unknown"
    Queued = "Queued"


class SerializerBackend(Enum):
    Dill = "Dill"
    ONNX = "ONNX"
    Custom = "Custom"
