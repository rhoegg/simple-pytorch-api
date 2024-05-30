import io
import torch
from pydantic import BaseModel

input_features = 48
output_features = 4

class NeuralNetLayerSpec(BaseModel):
    outputs: int
    parameters: dict

class NeuralNetModifierSpec(BaseModel):
    name: str
    parameters: dict

class NeuralNetSpec(BaseModel):
    layers: list[NeuralNetLayerSpec]
    modifiers: list[NeuralNetModifierSpec]

class NeuralNet(torch.nn.Module):
    def __init__(self, nn_spec: NeuralNetSpec):
        super(NeuralNet, self).__init__()
        
        self.spec = nn_spec
        self.layer_stack = torch.nn.Sequential()
        layer_input_features = input_features
        for layer in nn_spec.layers:
            self.layer_stack.append(torch.nn.Linear(layer_input_features, layer.outputs))
            layer_input_features = layer.outputs
            self.layer_stack.append(parse_activation(layer))
        self.layer_stack.append(torch.nn.Linear(layer_input_features, output_features))
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, amsgrad=True)
        # output_loss_delta = next(
        #     (m.parameters.delta for m in nn_spec.modifiers if m.parameters.has_key("delta")),
        #     1.0)
        # print("huber loss delta = " + output_loss_delta)

    @classmethod
    def from_saved(cls, data: bytes):
        return torch.load(data)
    
    def export(self):
        #onnx_program = torch.onnx.dynamo_export(m, torch.rand((1, input_features), dtype=torch.float32))
        with io.BytesIO() as buf:
            torch.save(self, buf)
            buf.seek(0)
            return buf.read()

    def forward(self, inputs):
        return self.layer_stack(inputs)

def parse_activation(layer_spec: NeuralNetLayerSpec):
    if "activation" not in layer_spec.parameters:
        return torch.nn.Identity()
    match layer_spec.parameters.get("activation"):
        case "relu":
            return torch.nn.ReLU()
        case "sigmoid":
            return torch.nn.Sigmoid()
        case "softmax":
            return torch.nn.Softmax()
        case "identity":
            return torch.nn.Identity()
        case "relu6":
            return torch.nn.ReLU6()
        case "celu":
            return torch.nn.CELU()
        case "elu":
            return torch.nn.ELU() # alpha param?
        case "tanh":
            return torch.nn.Tanh()
        case "mish":
            return torch.nn.Mish()
        case "softplus":
            return torch.nn.Softplus() # beta, threshold?
        case "hardswish":
            return torch.nn.Hardswish()
        case "gelu":
            return torch.nn.GELU()
        case "rrelu":
            return torch.nn.RReLU()
        case "hardtanh":
            return torch.nn.Hardtanh()
        case "tanhshrink":
            return torch.nn.Tanhshrink()
        case "hardsigmoid":
            return torch.nn.Hardsigmoid()
    return torch.nn.Identity()