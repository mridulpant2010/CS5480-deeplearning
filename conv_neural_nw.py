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