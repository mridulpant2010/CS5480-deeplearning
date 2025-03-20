import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
#TODO: understand the use of the batch size in above code.
#todo: criteria for deciding the number of neurons for each layer in a neural network.

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer (28x28 pixels) to hidden layer (128 neurons)
        self.fc2 = nn.Linear(128, 64)       # Hidden layer (128 neurons) to hidden layer (64 neurons)
        self.fc3 = nn.Linear(64, 10)        # Hidden layer (64 neurons) to output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # Flatten the input tensor
        x = torch.relu(self.fc1(x))        # Apply ReLU activation after first layer
        x = torch.relu(self.fc2(x))        # Apply ReLU activation after second layer
        x = self.fc3(x)                     # Output layer (logits)
        return x


model = MLP()
criterion = nn.CrossEntropyLoss()                  # Loss function for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent optimizer

n_epochs = 3

for epoch in range(n_epochs):
    model.train()                                  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()                      # Zero gradients from previous step
        output = model(data)                       # Forward pass
        loss = criterion(output, target)           # Compute loss
        loss.backward()                            # Backward pass (compute gradients)
        optimizer.step()                           # Update weights

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
model.eval()                                       # Set model to evaluation mode
test_loss = 0                                      # Initialize test loss variable
correct = 0                                        # Initialize correct predictions counter

with torch.no_grad():                              # Disable gradient calculation for evaluation
    for data, target in test_loader:
        output = model(data)                       # Forward pass on test data
        test_loss += criterion(output, target).item() * data.size(0)   # Accumulate loss
        pred = output.argmax(dim=1)                # Get predicted classes
        correct += pred.eq(target.view_as(pred)).sum().item()           # Count correct predictions

test_loss /= len(test_loader.dataset)              # Average test loss over dataset size
accuracy = correct / len(test_loader.dataset)      # Calculate accuracy

print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')


# Visualize some test results 
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Get predictions from the model 
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

# Plot some images with their predicted labels 
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f'Predicted: {predicted[i].item()}')
    plt.axis('off')
plt.show()
