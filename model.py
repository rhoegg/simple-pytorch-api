from pydantic import BaseModel

class NeuralNetSpec(BaseModel):
    layers: list
    modifiers: list

class NeuralNetLayerSpec(BaseModel):
    outputs: int
    parameters: dict

class NeuralNetModifierSpec(BaseModel):
    name: str
    parameters: dict