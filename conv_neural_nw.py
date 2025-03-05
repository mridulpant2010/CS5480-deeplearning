import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense
import kagglehub
import h5py,os
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
'''
• A3Dconvolution layer with 16 filters of size 5 ×5×5 and ReLU activation.
 • AMax Pooling 3D layer of size 2×2×2 with stride 2.
 • A3Dconvolution layer with 32 filters of size 3 ×3×3 and ReLU activation.
 • AMax Pooling 3D layer of size 2×2×2 with stride 2.
 • AGlobal Average Pooling (GAP) layer.
 • AnMLPwithonehidden layer, where the input size matches the output of the GAP layer and
 the output size is 10 (one for each digit class). The hidden layer should use ReLU activation,
 and the output layer should apply softmax activation.
'''

# model  = Sequential([
#     Conv3D(16,(5,5,5),activation='relu', input_shape=(64,64,64,1)),
#     MaxPooling3D(pool_size=(2,2,2),strides=2),
#     Conv3D(32,(3,3,3),activation='relu'),
#     MaxPooling3D(pool_size=(2,2,2),strides=2),
#     GlobalAveragePooling3D(),
#     Dense(128,activation='relu'),
#     Dense(10,activation='softmax')
# ])

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()

#setx KAGGLE_CONFIG_DIR "C:\Users\pantm\downloads"
#os.environ['KAGGLE_CONFIG_DIR'] 
custom_path = "C:\\Users\\pantm\\Downloads\\kagglehub\\full_dataset_vectors.h5"

# Download latest version

#h5py_path = kagglehub.dataset_download("daavoo/3d-mnist")

#print("Path to dataset files:", h5py_path)

with h5py.File(custom_path,'r') as h5_file:
    #image_dataset= h5_file['image_dataset'][:]
    X_train = h5_file["X_train"][:]
    y_train = h5_file["y_train"][:]    
    X_test = h5_file["X_test"][:] 
    y_test = h5_file["y_test"][:]
  

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)  

print(X_train)

# X_train = X_train.reshape(-1, 64, 64, 64, 1).astype('float32') / 255
# X_test = X_test.reshape(-1, 64, 64, 64, 1).astype('float32') / 255
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_accuracy:.4f}")

# how do i plot the 3d mnist dataset,

with h5py.File("C:\\Users\\pantm\\Downloads\\kagglehub\\train_point_clouds.h5", "r") as points_dataset:        
    digits = []
    for i in range(10):
        digit = (points_dataset[str(i)]["img"][:], 
                 points_dataset[str(i)]["points"][:], 
                 points_dataset[str(i)].attrs["label"]) 
        digits.append(digit)
        
x_c = [r[0] for r in digits[0][1]]
y_c = [r[1] for r in digits[0][1]]
z_c = [r[2] for r in digits[0][1]]
trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))

data = [trace1]
layout = go.Layout(height=500, width=600, title= "Digit: "+str(digits[0][2]) + " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


import numpy as np
def sigmoid(X, weights, bias):
    z = X.dot(weights) + bias
    return 1 / (1 + np.exp(-z))


def gradient_descent_learning(X, y, weights, bias, eta=0.01, n_iter=1000):
    m = X.shape[0]  # Number of samples
    for _ in range(n_iter):
        predictions = sigmoid(X, weights, bias)
        errors = predictions - y
        gd_weights = X.T.dot(errors) / m
        gd_bias = np.sum(errors) / m

        weights -= eta * gd_weights
        bias -= eta * gd_bias

    return weights, bias


def cross_entropy_loss(X, y, weights, bias):
    m = len(y)
    h = sigmoid(X, weights, bias)
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    return cost


import numpy as np

class Conv3D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size, filter_size) / (filter_size ** 1.5)
        self.bias = np.zeros((num_filters, 1))

    def forward(self, input):
        self.input = input
        self.output = np.zeros((input.shape[0] - self.filter_size + 1,
                                input.shape[1] - self.filter_size + 1,
                                input.shape[2] - self.filter_size + 1,
                                self.num_filters))
        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[2]):
                    input_slice = input[i:i+self.filter_size, j:j+self.filter_size, k:k+self.filter_size]
                    for f in range(self.num_filters):
                        self.output[i, j, k, f] = np.sum(input_slice * self.filters[f]) + self.bias[f]
        return self.output

class MaxPool3D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        self.output = np.zeros((
            (input.shape[0] - self.pool_size) // self.stride + 1,
            (input.shape[1] - self.pool_size) // self.stride + 1,
            (input.shape[2] - self.pool_size) // self.stride + 1,
            input.shape[3]
        ))
        for i in range(0, self.output.shape[0], self.stride):
            for j in range(0, self.output.shape[1], self.stride):
                for k in range(0, self.output.shape[2], self.stride):
                    self.output[i//self.stride, j//self.stride, k//self.stride] = np.max(
                        input[i:i+self.pool_size, j:j+self.pool_size, k:k+self.pool_size], axis=(0,1,2)
                    )
        return self.output

class GlobalAvgPool3D:
    def forward(self, input):
        return np.mean(input, axis=(0,1,2))

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

class CNN3D:
    def __init__(self):
        self.conv1 = Conv3D(16, 5)
        self.pool1 = MaxPool3D(2, 2)
        self.conv2 = Conv3D(32, 3)
        self.pool2 = MaxPool3D(2, 2)
        self.gap = GlobalAvgPool3D()
        self.fc1 = Dense(32, 64)
        self.fc2 = Dense(64, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        x = self.gap.forward(x)
        x = self.fc1.forward(x)
        x = relu(x)
        x = self.fc2.forward(x)
        return softmax(x)

# Create and use the model
model = CNN3D()
input_data = np.random.randn(28, 28, 28, 1)  # Example input
output = model.forward(input_data)
print(output)

import numpy as np
## Introduce the channel dimention in the input dataset 
xtrain = np.ndarray((X_train.shape[0], 4096, 1))
xtest = np.ndarray((X_test.shape[0], 4096, 1))
xtrain.shape
## convert to 1 + 4D space (1st argument represents number of rows in the dataset)
xtrain = xtrain.reshape(X_train.shape[0], 16, 16, 16, 1)
xtest = xtest.reshape(X_test.shape[0], 16, 16, 16, 1)
xtrain.shape
# what is the size of the input tensor?
input_volume = xtrain[0]


