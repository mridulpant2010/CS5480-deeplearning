import torch
import torch.nn as nn

tensor1 = torch.tensor([3,4,5])
print("tensor from the list, ",tensor1)

tensor2 = torch.zeros(2,3)
print("tensor of the zeroes, ",tensor2)

tensor3 = torch.rand(2,3)
print("tensor of the random numbers, ",tensor3)


m = nn.Linear(20, 30) 
input = torch.randn(128, 20)
output = m(input)
# in this code nn.Linear is instantiated and m object is created.
# then why do m is instantiated, what is this phenomena in python called.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,1)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
model = SimpleNN()

import torch.optim as optim

# Sample data
x_train = torch.randn(100, 10)  # 100 samples, 10 features each
y_train = torch.randn(100, 1)    # Corresponding labels

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # Number of epochs
    model.train()         # Set model to training mode
    
    optimizer.zero_grad() # Zero gradients from previous step
    outputs = model(x_train) # Forward pass
    loss = criterion(outputs, y_train) # Compute loss
    
    loss.backward()       # Backward pass (compute gradients)
    optimizer.step()      # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


x_test = torch.randn(20, 10)   # 20 samples for testing
y_test = torch.randn(20, 1)
model.eval()               # Set model to evaluation mode
with torch.no_grad():      # Disable gradient calculation for evaluation
    test_outputs = model(x_test) # Forward pass on test data
    test_loss = criterion(test_outputs, y_test) # Compute loss

print(f'Test Loss: {test_loss.item()}')