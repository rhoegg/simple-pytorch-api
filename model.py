import io
from pydantic import BaseModel
from torch import nn, load, save

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

class NeuralNet(nn.Module):
    def __init__(self, nn_spec: NeuralNetSpec):
        super(NeuralNet, self).__init__()
        
        self.spec = nn_spec
        self.layer_stack = nn.Sequential()
        layer_input_features = input_features
        for layer in nn_spec.layers:
            self.layer_stack.append(nn.Linear(layer_input_features, layer.outputs))
            layer_input_features = layer.outputs
            self.layer_stack.append(parse_activation(layer))
        self.layer_stack.append(nn.Linear(layer_input_features, output_features))

        # output_loss_delta = next(
        #     (m.parameters.delta for m in nn_spec.modifiers if m.parameters.has_key("delta")),
        #     1.0)
        # print("huber loss delta = " + output_loss_delta)

    @classmethod
    def from_saved(cls, data: bytes):
        return load(data)
    
    def export(self):
        #onnx_program = torch.onnx.dynamo_export(m, torch.rand((1, input_features), dtype=torch.float32))
        with io.BytesIO() as buf:
            save(self, buf)
            buf.seek(0)
            return buf.read()

    def forward(self, inputs):
        return self.layer_stack(inputs)

def parse_activation(layer_spec: NeuralNetLayerSpec):
    if "activation" not in layer_spec.parameters:
        return nn.Identity()
    match layer_spec.parameters.get("activation"):
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "softmax":
            return nn.Softmax()
        case "identity":
            return nn.Identity()
        case "relu6":
            return nn.ReLU6()
        case "celu":
            return nn.CELU()
        case "elu":
            return nn.ELU() # alpha param?
        case "tanh":
            return nn.Tanh()
        case "mish":
            return nn.Mish()
        case "softplus":
            return nn.Softplus() # beta, threshold?
        case "hardswish":
            return nn.Hardswish()
        case "gelu":
            return nn.GELU()
        case "rrelu":
            return nn.RReLU()
        case "hardtanh":
            return nn.Hardtanh()
        case "tanhshrink":
            return nn.Tanhshrink()
        case "hardsigmoid":
            return nn.Hardsigmoid()
    return nn.Identity()