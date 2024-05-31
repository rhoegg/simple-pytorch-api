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
        self.huber_loss_delta = parse_huber_loss_delta(nn_spec)
        self.learning_rate = parse_learning_rate(nn_spec)
        self.weight_decay = parse_weight_decay(nn_spec)

    @classmethod
    def from_saved(cls, data: bytes):
        return torch.load(data)
    
    def export(self):
        with io.BytesIO() as buf:
            torch.save(self, buf)
            buf.seek(0)
            return buf.read()

    def forward(self, inputs):
        return self.layer_stack(inputs)
    
    def optimizer(self):
        if "updater" not in (m.name for m in self.spec.modifiers):
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True)
        mod = next((m for m in self.spec.modifiers if m.name == "updater"))
        match mod.parameters.get("name"):
            # TODO: more params on each of these
            case "sgd":
                return torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "nadam":
                return torch.optim.NAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 
            case "adam":
                return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "adagrad":
                return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "adamax":
                return torch.optim.Adamax(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "nesterovs":
                return torch.optim.SGD(self.parameters(), lr=self.learning_rate, nesterov=True, weight_decay=self.weight_decay)
            case "adadelta":
                return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            case "amsgrad":
                return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

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

def parse_huber_loss_delta(nn_spec: NeuralNetSpec) -> float:
    return next(
        (m.parameters["delta"] for m in nn_spec.modifiers if m.name=="huberLoss" and "delta" in m.parameters),
        1.0
    )

def parse_learning_rate(nn_spec: NeuralNetSpec) -> float:
    return next(
        (m.parameters["factor"] for m in nn_spec.modifiers if m.name=="learningRate" and "factor" in m.parameters),
        0.001
    )

def parse_weight_decay(nn_spec: NeuralNetSpec) -> float:
    return next(
        (m.parameters["coefficient"] for m in nn_spec.modifiers if m.name=="weightDecay" and "coefficient" in m.parameters),
        0
    )    

def train_model(model: NeuralNet, features: list[float], labelled: list[float], device):
    criterion = torch.nn.HuberLoss(delta=model.huber_loss_delta)
    
    t_features = torch.tensor(data=features, dtype=torch.float32).to(device)
    t_labelled = torch.tensor(data=labelled, dtype=torch.float32).to(device)
	
    model.optimizer().zero_grad()
    predicted = model(t_features)
    print(predicted)
    loss = criterion(predicted, t_labelled)
    loss.backward()
    model.optimizer().step()
