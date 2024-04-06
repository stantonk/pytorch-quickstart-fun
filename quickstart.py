# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from lib import get_device

TRAIN = False

### Working with data

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
#
# # Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

### Creating Models


# Get cpu, gpu or mps device for training.
device = get_device()


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

### Optimizing the Model Parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


### TRAINING
if TRAIN:
    epochs = 54
    # learning rate started slowing down a lot at Epoch 54
    # it did keep improving at least up to Epoch 79
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    ### Saving Models
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")




### Loading Models

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

### Make predictions / inference

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

class_to_index = {}
for i, c in enumerate(classes):
    class_to_index[c] = i

predictions, correct_predictions = 0, 0
model.eval()
# x, y = test_data[0][0], test_data[0][1]
print(f'testing the model against test_data set:')
for x, y in test_data:
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        predictions += 1
        if predicted == actual:
            correct_predictions += 1
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')

print(f'predictions = {predictions} correct_predictions = {correct_predictions}')
print(f'accuracy = {(correct_predictions / float(predictions)) * 100.0}%')

from PIL import Image
raw_images_and_labels = [
    ('./testimg1.png', class_to_index['T-shirt/top']),
    ('./testimg1-white.png', class_to_index['T-shirt/top']),
    ('./testimg2.png', class_to_index['T-shirt/top']),
    ('./testimg2-white.png', class_to_index['T-shirt/top']),
    ('./testimg3.png', class_to_index['T-shirt/top']),
    ('./testimg3-white.png', class_to_index['T-shirt/top']),
    ('./testimg4.png', class_to_index['Sneaker']),
    ('./testimg4-white.png', class_to_index['Sneaker']),
]

images_and_labels = []
for img_path, class_index in raw_images_and_labels:
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((28, 28))
    # img.show()
    images_and_labels.append((img, class_index))

with torch.no_grad():
    fig = plt.figure(figsize=(20, 7))
    for i, (img, class_index) in enumerate(images_and_labels):
        ax = fig.add_subplot(4, 2, i + 1, xticks=[], yticks=[])
        imgTensor = ToTensor()(img)
        imgTensor = imgTensor.to(device)
        # print(imgTensor.shape)
        # print(np.squeeze(imgTensor).shape)
        plt.imshow(np.squeeze(imgTensor))
        pred = model(imgTensor)
        predicted, actual = pred[0].argmax(0), class_index
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        color = 'green' if classes[predicted] == classes[class_index] else 'red'
        ax.set_title(f'predicted {classes[predicted]} actual {classes[class_index]}', color=color)
    plt.show()
