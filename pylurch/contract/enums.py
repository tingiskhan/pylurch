from enum import Enum


class Status(Enum):
    Running = "Running"
    Done = "Done"
    Failed = "Failed"
    Cancelled = "Cancelled"
    Unknown = "Unknown"
    Queued = "Queued"


class Backend(Enum):
    Sklearn = "Sklearn"
    Pytorch = "Pytorch"
    Dill = "Dill"
    ONNX = "ONNX"
    Custom = "Custom"


class ArtifactType(Enum):
    Model = "Model"
    State = "State"
