from enum import Enum


def chunk(iterable, cs):
    for i in range(0, len(iterable), cs):
        yield iterable[i:i + cs]


class Constants(Enum):
    ChunkSize = 999