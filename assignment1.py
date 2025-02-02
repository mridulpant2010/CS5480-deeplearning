# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# num_points = 1000
# np.random.seed(42)

# # Generate random points
# class_1 = np.random.randn(num_points // 2, 2) + [2, 2]
# class_2 = np.random.randn(num_points // 2, 2) + [-2, -2]

# # Combine the points
# data = np.vstack((class_1, class_2))
# labels = np.hstack((np.zeros(num_points // 2), np.ones(num_points // 2)))

# # Plot the dataset
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='bwr', edgecolors='k')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Linearly Separable Dataset')
# plt.show()
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


class Perceptron:
    def __init__(self,iter):
        self.weights = None
        self.iter = iter
    
    def pla_train(self,X,y):
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        # what shall be the convergence criteria?
        k = 0
        while k<self.iter:
            for i in range(n_samples):
                linear_output = (np.transpose(self.weights) @ X[i])*y[i]
                if linear_output <=0:
                    self.weights = self.weights+ y[i]*X[i]
            k+=1
    
    def pla_predict(self,X):
        linear_output = np.dot(X[i],self.weights)
        return linear_output


if __name__ == "__main__":

    X,y =datasets.make_blobs(n_samples=1000,n_features=2,centers=2,center_box=(0,10))
    y = np.where(y==0,-1,1)
    # print(X,y)
    # print(X[:,0],X[:,1])
    def visualize(X,y,msg):
        plt.scatter(X[:,0],X[:,1], c=y, cmap='bwr', edgecolors='k')
        plt.xlabel(f'Feature 1 {msg}')
        plt.ylabel(f'Feature 2 {msg}')
        plt.title(f'Linearly Separable Dataset {msg}')
        plt.show()
        
    perceptron = Perceptron(1000) # criteria to chose the hyperparameter
    perceptron.pla_train(X,y)
    print(perceptron.weights)


    # perceptron2 = Perceptron(1000) # criteria to chose the hyperparameter
    # perceptron2.pla_train(X,y)
    # print(perceptron2.weights)
    
    # perceptron3 = Perceptron(10) # criteria to chose the hyperparameter
    # perceptron3.pla_train(X,y)
    # print(perceptron3.weights)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Plot decision boundary
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = -(perceptron.weights[0] * x1 ) / perceptron.weights[1]
    plt.plot(x1, x2, color='black')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Classifier with Decision Boundary')
    plt.show()

    #? why so much inconsistency with the data. 
# def perceptron_learning_algorithm(x,y):
#     k = 0
#     w_k=np.zeros(x.shape)
#     max_iterations = min(1000,len(y))
#     while k<max_iterations :
#         if y[k]*((np.transpose(w_k))*x[k])<=0:
#             w_k = w_k + y[k]*x[k]
#             k+=1
            
    # how shall i fit this algo for now. i think