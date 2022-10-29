from typing import List, Type, Dict

from oba import Obj

from model.base_model import BaseModel
from model.matching.nrms_model import NRMSModel

MODEL_LIST = [
    NRMSModel,
]  # type: List[Type[BaseModel]]
MODELS = {model.__name__: model for model in MODEL_LIST}  # type: Dict[str, Type[BaseModel]]


def parse(model):
    if model.name not in MODELS:
        raise ValueError(f'No matched model: {model.name}')

    model_class = MODELS[model.name]
    config = dict()
    if model.config:
        config = Obj.raw(model.config)
    model_config = model_class.config_class(**config)

    return model_class(config=model_config)
