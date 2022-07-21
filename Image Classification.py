import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import numpy as np

# Creating the dataset
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())
test_dataset = MNIST(root='data/',
                     train=False,
                     transform=transforms.ToTensor())


# Function for splitting and shuffling the datasets
def split_indices(n, val_pct):
    # Size of validation set
    n_val = int(val_pct*n)
    # Random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # First n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# Calling the function
train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)

# Batch size
batch_size = 100

# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset,
                          batch_size,
                          sampler=train_sampler)
# Validation sampler and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset,
                        batch_size,
                        sampler=val_sampler)

# Defining the inputs
input_size = 28*28
num_classes = 10


# Logistic regression model
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out


# Defining the model
model = MnistModel()

for images, labels in train_loader:
    outputs = model(images)
    break

# Applying softmax for each output row
probs = F.softmax(outputs, dim=1)
max_probs, preds = torch.max(probs, dim=1)

# Loss for current batch of data
loss_fn = F.cross_entropy
loss = loss_fn(outputs, labels)

# Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Function for calculating the loss and applying the optimizer
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    # Calculate loss
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        # Computing gradients
        loss.backward()
        # Updating parameters
        opt.step()
        # Resetting gradients
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # Computing the metric
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


# Evaluation function
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # Passing each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb in valid_dl]

        # Separating losses, counts and metrics
        losses, nums, metrics = zip(*results)

        # Total size of the dataset
        total = np.sum(nums)

        # Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # Avg. of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric


# Accuracy of the model
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Checking the loss and accuracy of the untrained model
val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))


# Main training loop
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        # Training
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        # Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # Printing progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))


# Model training
# fit(15, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# Saving the model
# torch.save(model.state_dict(), 'mnist-logistics.pth')

# Loading already trained model
model2 = MnistModel()
model2.load_state_dict(torch.load('mnist-logistics.pth'))

# Checking the loss and accuracy of the loaded model
val_loss, total, val_acc = evaluate(model2, loss_fn, val_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))


# Image prediction function using the model
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


# Checking the trained model with test dataset
img, label = test_dataset[0]
print('Label:', label, ',Predicted:', predict_image(img, model2))
