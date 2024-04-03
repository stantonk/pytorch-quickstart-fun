# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# Relevant Reading
# ReLU: https://arxiv.org/abs/1603.05201; https://arxiv.org/pdf/1603.05201.pdf
# CNNs: https://en.wikipedia.org/wiki/Convolutional_neural_network
# CNN vs. LLM: https://hansheng0512.medium.com/understanding-the-difference-between-llms-and-cnns-in-machine-learning-16ed76f1965f

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # nn.Sequential is an ordered container of modules. The data is passed
        # through all the modules in the same order as defined.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# Calling the model on the input returns a 2-dimensional tensor with dim=0
# corresponding to each output of 10 raw predicted values for each class, and
# dim=1 corresponding to the individual values of each output. We get the
# prediction probabilities by passing it through an instance of the nn.Softmax
# module.
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
# The last linear layer of the neural network returns logits - raw values in
# [-infty, infty] - which are passed to the nn.Softmax module. The logits are
# scaled to values [0, 1] representing the model’s predicted probabilities
# for each class. dim parameter indicates the dimension along which the
# values must sum to 1.
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# Many layers inside a neural network are parameterized, i.e. have associated
# weights and biases that are optimized during training. Subclassing nn.Module
# automatically tracks all fields defined inside your model object, and makes
# all parameters accessible using your model’s parameters() or named_parameter
# () methods.

# In this example, we iterate over each parameter, and print its size and a
# preview of its values.
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")