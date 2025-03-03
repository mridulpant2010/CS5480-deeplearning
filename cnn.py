import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Conv3D_FromScratch:
    def __init__(self, input_shape, kernel_size, num_filters, stride=1, padding=0):
        """
        Initialize a 3D convolution layer from scratch
        
        Args:
            input_shape: Tuple (channels, depth, height, width)
            kernel_size: Int or tuple for kernel dimensions
            num_filters: Number of filters to apply
            stride: Int or tuple for stride in each dimension
            padding: Int or tuple for padding in each dimension
        """
        self.input_channels, self.input_depth, self.input_height, self.input_width = input_shape
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
            
        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding
            
        self.num_filters = num_filters
        
        # Initialize kernels and biases with small random values
        self.kernels = np.random.randn(
            num_filters, 
            self.input_channels, 
            self.kernel_size[0], 
            self.kernel_size[1], 
            self.kernel_size[2]
        ) * 0.01
        
        self.biases = np.zeros(num_filters)
        
        # Initialize gradients
        self.dkernels = np.zeros_like(self.kernels)
        self.dbiases = np.zeros_like(self.biases)
        
        # Calculate output dimensions
        self.output_depth = (self.input_depth + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        self.output_height = (self.input_height + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        self.output_width = (self.input_width + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1
        
        # Store the input and output for backpropagation
        self.input = None
        self.output = None
        
    def pad_input(self, input_data):
        """Add zero padding to the input"""
        # If no padding, return the original input
        if all(p == 0 for p in self.padding):
            return input_data
            
        # Get dimensions
        batch_size, channels, depth, height, width = input_data.shape
        
        # Create a padded output
        padded_output = np.zeros((
            batch_size,
            channels,
            depth + 2 * self.padding[0],
            height + 2 * self.padding[1],
            width + 2 * self.padding[2]
        ))
        
        # Fill the padded output with the input data
        padded_output[:, :,
                      self.padding[0]:self.padding[0] + depth,
                      self.padding[1]:self.padding[1] + height,
                      self.padding[2]:self.padding[2] + width] = input_data
                      
        return padded_output
    
    def forward(self, input_data):
        """
        Perform a forward pass of the 3D convolution operation
        
        Args:
            input_data: Input data of shape (batch_size, channels, depth, height, width)
            
        Returns:
            output: Convolution result of shape (batch_size, num_filters, output_depth, output_height, output_width)
        """
        # Store the input for backpropagation
        self.input = input_data
        
        # Get dimensions
        batch_size = input_data.shape[0]
        
        # Pad the input
        padded_input = self.pad_input(input_data)
        
        # Initialize output
        output = np.zeros((batch_size, self.num_filters, self.output_depth, self.output_height, self.output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for d in range(self.output_depth):
                    for h in range(self.output_height):
                        for w in range(self.output_width):
                            # Calculate the start indices for this position
                            d_start = d * self.stride[0]
                            h_start = h * self.stride[1]
                            w_start = w * self.stride[2]
                            
                            # Calculate the end indices for this position
                            d_end = d_start + self.kernel_size[0]
                            h_end = h_start + self.kernel_size[1]
                            w_end = w_start + self.kernel_size[2]
                            
                            # Extract the region to apply the filter to
                            region = padded_input[b, :, d_start:d_end, h_start:h_end, w_start:w_end]
                            
                            # Apply the filter (element-wise multiplication and sum)
                            output[b, f, d, h, w] = np.sum(region * self.kernels[f]) + self.biases[f]
        
        # Store the output for backpropagation
        self.output = output
        
        return output
    
    def backward(self, doutput, learning_rate):
        """
        Perform a backward pass of the 3D convolution operation
        
        Args:
            doutput: Gradient of the loss with respect to the output
            learning_rate: Learning rate for gradient descent
            
        Returns:
            dinput: Gradient of the loss with respect to the input
        """
        # Initialize gradients
        batch_size = self.input.shape[0]
        dinput = np.zeros_like(self.input)
        self.dkernels = np.zeros_like(self.kernels)
        self.dbiases = np.zeros_like(self.biases)
        
        # Pad the input
        padded_input = self.pad_input(self.input)
        
        # Create a padded dinput
        padded_dinput = np.zeros_like(padded_input)
        
        # Compute gradients for biases
        for f in range(self.num_filters):
            self.dbiases[f] = np.sum(doutput[:, f, :, :, :])
        
        # Compute gradients for kernels and input
        for b in range(batch_size):
            for f in range(self.num_filters):
                for d in range(self.output_depth):
                    for h in range(self.output_height):
                        for w in range(self.output_width):
                            # Calculate the start indices for this position
                            d_start = d * self.stride[0]
                            h_start = h * self.stride[1]
                            w_start = w * self.stride[2]
                            
                            # Calculate the end indices for this position
                            d_end = d_start + self.kernel_size[0]
                            h_end = h_start + self.kernel_size[1]
                            w_end = w_start + self.kernel_size[2]
                            
                            # Extract the region to apply the filter to
                            region = padded_input[b, :, d_start:d_end, h_start:h_end, w_start:w_end]
                            
                            # Compute gradient of kernels
                            self.dkernels[f] += region * doutput[b, f, d, h, w]
                            
                            # Compute gradient of input
                            padded_dinput[b, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
                                self.kernels[f] * doutput[b, f, d, h, w]
        
        # Extract the center part of padded_dinput
        if all(p == 0 for p in self.padding):
            dinput = padded_dinput
        else:
            dinput = padded_dinput[:, :,
                                   self.padding[0]:self.padding[0] + self.input_depth,
                                   self.padding[1]:self.padding[1] + self.input_height,
                                   self.padding[2]:self.padding[2] + self.input_width]
        
        # Update parameters
        self.kernels -= learning_rate * self.dkernels
        self.biases -= learning_rate * self.dbiases
        
        return dinput

# Activation functions
class Activation:
    def __init__(self, activation_type="relu"):
        """
        Initialize an activation function
        
        Args:
            activation_type: Type of activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
        """
        self.activation_type = activation_type
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        """
        Apply the activation function
        
        Args:
            input_data: Input data
            
        Returns:
            output: Activated output
        """
        self.input = input_data
        
        if self.activation_type == "relu":
            self.output = np.maximum(0, input_data)
        elif self.activation_type == "sigmoid":
            self.output = 1 / (1 + np.exp(-input_data))
        elif self.activation_type == "tanh":
            self.output = np.tanh(input_data)
        elif self.activation_type == "leaky_relu":
            self.output = np.where(input_data > 0, input_data, 0.01 * input_data)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
            
        return self.output
    
    def backward(self, doutput):
        """
        Compute the gradient of the activation function
        
        Args:
            doutput: Gradient of the loss with respect to the output
            
        Returns:
            dinput: Gradient of the loss with respect to the input
        """
        if self.activation_type == "relu":
            dinput = np.array(doutput, copy=True)
            dinput[self.input <= 0] = 0
        elif self.activation_type == "sigmoid":
            s = 1 / (1 + np.exp(-self.input))
            dinput = doutput * s * (1 - s)
        elif self.activation_type == "tanh":
            dinput = doutput * (1 - np.tanh(self.input) ** 2)
        elif self.activation_type == "leaky_relu":
            dinput = np.array(doutput, copy=True)
            dinput[self.input <= 0] *= 0.01
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
            
        return dinput

# MaxPooling layer
class MaxPool3D:
    def __init__(self, pool_size=2, stride=None):
        """
        Initialize a 3D max pooling layer
        
        Args:
            pool_size: Size of the pooling window
            stride: Stride of the pooling operation (defaults to pool_size)
        """
        # Handle pool_size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        # Handle stride
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
            
        self.input = None
        self.output = None
        self.max_indices = None
    
    def forward(self, input_data):
        """
        Perform a forward pass of the 3D max pooling operation
        
        Args:
            input_data: Input data of shape (batch_size, channels, depth, height, width)
            
        Returns:
            output: Max pooled output
        """
        self.input = input_data
        
        # Get dimensions
        batch_size, channels, depth, height, width = input_data.shape
        
        # Calculate output dimensions
        output_depth = (depth - self.pool_size[0]) // self.stride[0] + 1
        output_height = (height - self.pool_size[1]) // self.stride[1] + 1
        output_width = (width - self.pool_size[2]) // self.stride[2] + 1
        
        # Initialize output and max_indices
        output = np.zeros((batch_size, channels, output_depth, output_height, output_width))
        self.max_indices = np.zeros((batch_size, channels, output_depth, output_height, output_width, 3), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for d in range(output_depth):
                    for h in range(output_height):
                        for w in range(output_width):
                            # Calculate the start indices for this position
                            d_start = d * self.stride[0]
                            h_start = h * self.stride[1]
                            w_start = w * self.stride[2]
                            
                            # Calculate the end indices for this position
                            d_end = min(d_start + self.pool_size[0], depth)
                            h_end = min(h_start + self.pool_size[1], height)
                            w_end = min(w_start + self.pool_size[2], width)
                            
                            # Extract the region to apply the pooling to
                            region = input_data[b, c, d_start:d_end, h_start:h_end, w_start:w_end]
                            
                            # Find the maximum value and its index
                            max_val = np.max(region)
                            max_index = np.unravel_index(np.argmax(region), region.shape)
                            
                            # Store the maximum value and its index
                            output[b, c, d, h, w] = max_val
                            self.max_indices[b, c, d, h, w] = np.array([d_start + max_index[0], 
                                                                       h_start + max_index[1], 
                                                                       w_start + max_index[2]])
        
        self.output = output
        
        return output
    
    def backward(self, doutput):
        """
        Perform a backward pass of the 3D max pooling operation
        
        Args:
            doutput: Gradient of the loss with respect to the output
            
        Returns:
            dinput: Gradient of the loss with respect to the input
        """
        # Initialize dinput
        dinput = np.zeros_like(self.input)
        
        # Get dimensions
        batch_size, channels, output_depth, output_height, output_width = doutput.shape
        
        # Distribute gradients to the max elements
        for b in range(batch_size):
            for c in range(channels):
                for d in range(output_depth):
                    for h in range(output_height):
                        for w in range(output