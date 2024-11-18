from enum import Enum
from pydantic import BaseModel


class Classification(Enum):
    DEGRADE = "degrade"
    DEGRADANT = "degradant"
    AUCUN_RAPPORT = "aucun-rapport"


class ResponseSchemas(BaseModel):
    classname: Classification
    probability: float
    mention: str | None