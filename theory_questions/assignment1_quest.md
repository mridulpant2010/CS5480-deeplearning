Based on the provided MLP implementation code, here are 30 questions divided into easy, medium, and hard categories:

## Easy Questions

1. What activation function is used in the hidden layer of this MLP implementation?
2. How many layers does this MLP architecture have?
3. What are the two loss functions implemented in this MLP class?
   1. mse vs mae, when to use and which?
      1. MSE = (1/n) * Σ(y_true - y_pred)²
         1. sensitive to outliers: it maximizes the loss by squaring and is sensitive to the extreme/large values
         2. smooth optimization: derivative is smooth and continuous, making it computationally efficient and easier to optimize with the gradient descent 
      2. MAE = (1/n) * Σ|y_true - y_pred|
         1. robust to outliers: it gives the median error and not the mean error, explain?
         2. MAE has a non-differentiable point at zero, which makes optimisation trickier for some of the algorithms. explain with an example.
      3. In application and in real life should we first be plotting the loss distribution first and then deciding based on if it is gradient (MSE) or laplacian (MAE)
         1. Train a simple model with MSE and plot a histogram of the residuals
         2. Fit both the gaussian and laplace distribution to the residuals and determine which distribution better fits the model.
4. What is the default weight initialization strategy in the MLP constructor?
   1. compare when to use the default weight initialization with the random weight initialization stratgey.
5. How many neurons are in the hidden layer of the first experiment?
6. What learning rate is used in most of the experiments?
7. What is the function used to generate synthetic data for training?
   1. need to review
8. How many samples are generated in the synthetic dataset?
9.  What preprocessing technique is applied to the input features before training?


## Medium Questions

1. Why might random weight initialization be preferred over zero initialization for neural networks?
   1. because zero initialization being static all the data points will have the same behavior
   2. Would the initialization strategy need to change for different activation functions (e.g., tanh, sigmoid)?
2. How does the backward_pass method implement gradient descent?
3. What is the purpose of the StandardScaler in the training process?
   1. good question , removing the mean (mean =0) and variance =1 and scaling to unit variance, normalization
   2. followup question -> how normalization helps in the convergence as the weights are standardized so oscillation will also be reduced.
      1. convergence of what? 
         1. gradient descent algorithm
4. How does the implementation handle the derivatives of different loss functions?
   1. can we have the derivative of the ReLU function.
   2. for the +ve x>0 -> f'(x) = x and if x<0 then f'(x)=0 as f(x)=0
5. What is the mathematical function used to generate the target values in the synthetic dataset?
6. How would changing the ReLU activation to tanh affect the training process?
   1. both provides the non-linearity , so how does it help?
   2. Tanh is zero-centered and provides stronger gradient than the sigmoid. How, explain.
   3. ReLU is computationally efficient and reduces the vanishing gradient issue. Explain
7. What modifications would be needed to implement mini-batch gradient descent?
   1. Don't know in detail.
8.  How might the choice between MSE and MAE loss affect the model's sensitivity to outliers?
9.  How might different random initialization strategies (e.g., Xavier/Glorot, He initialization) affect learning?
10. How does the code structure separate the forward and the backward passes?
11. What is the purpose of the keepdims parameter in the np.sum function during gradient calculation?
    
## Hard Questions

1. What potential issues might arise from using zero initialization for all weights in deep neural networks?
   1. symmetry breaking problem -> same output and same gradients during BP given by all the neurons , the updates happens identical for all the neurons across all layers 
   2. zero init experiment shows poorer convergence compared to random init expt. explain how?
   3. 
2. How would you modify this implementation to include regularization techniques like L1 or L2 regularization?
   1. prevents overfitting helps in generalization by adding penalties to the loss function during the training. It adds penalty but where
   2. regularization strength - > lambda = 0.01
   3. L1 regularization -> 
      1. Total loss = Original Loss + λ * Σ|w|
      2. criteria for choosing this hyperparameter lambda (λ), can you argue what all the characteristics we might get
   4. L2 regularization -> 
      1. original loss +  λ *(Σ|w|^2)
3. What changes would be required to implement a deeper network with multiple hidden layers?
4. How would you implement early stopping to prevent overfitting in this MLP implementation?
   1. why do we need early stopping, first of all.. Regularization technique, save all those weights that gave the better model performance.
   2. why validation set is important and what changes do you make? Do we apply all the regularization techniques after analyzing the validation set.
5. What modifications would be needed to handle classification problems instead of regression?
   1. loss function -> binary cross entropy / multi class entropy 
   2. activation function -> ReLU or tanh (hidden layers)
   3. outer layer - > softmax for k-class 
   4. metrics
      1. accuracy
      2. precision
      3. recall
      4. f1-score
      5. roc auc (binary classification)
   5. what are some metrics that tracks in case of the regression?
      1. loss functions: mse, mae, rmse, rmsle, r-squared
6. How could you implement momentum or adaptive learning rate methods in this gradient descent algorithm?
7. What are the computational bottlenecks in this implementation, and how could they be optimized?
   1. MLP implementation doesn't uses embeddings how does it helps?
   2. 
8. How would you implement dropout regularization in this MLP architecture? (Try this out, important)
   1. dropout is a deep regularization technique and prevents overfitting, so ?
9.  What changes would be required to make this implementation compatible with GPU acceleration?
10. How would you modify the code to implement batch normalization between layers?
    1.  don't remember what is the batch normalization technique?

Citations:
[1] https://machinelearninggeek.com/multi-layer-perceptron-neural-network-using-python/
[2] https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
[3] https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron
[4] https://programmer.ie/post/questions2/
[5] https://stackoverflow.com/questions/42092461/first-neural-network-mlp-from-scratch-python-questions
[6] https://github.com/KirillShmilovich/MLP-Neural-Network-From-Scratch/blob/master/MLP.ipynb
[7] https://datascience.stackexchange.com/questions/45480/coding-mlp-good-practices
[8] http://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation1/Your%20First%20MLP%20Code.pdf
[9] https://www.youtube.com/watch?v=ItkSCYzSD34
[10] https://www.studocu.com/in/document/shri-vishnu-engineering-college-for-women/machine-learning/mlp-question-bank/80639895
[11] https://www.reddit.com/r/learnmachinelearning/comments/lfzn24/neural_network_basic_questions_mostly_applicable/
[12] https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/exams/final_sol.pdf
[13] https://stackoverflow.com/questions/tagged/mlp
[14] https://egyankosh.ac.in/bitstream/123456789/12687/1/Unit-6.pdf
[15] https://stackoverflow.com/questions/53238319/multilayer-perceptron-questions
[16] https://www.sanfoundry.com/neural-networks-questions-answers-multi-layer-feedforward-neural-network/
[17] https://www.101computing.net/python-challenges-intermediate-level/
[18] https://www.youtube.com/watch?v=HOaq2-symDw
[19] https://github.com/rcassani/mlp-example
[20] https://www.allaboutcircuits.com/technical-articles/advanced-machine-learning-with-the-multilayer-perceptron/
[21] https://www.youtube.com/watch?v=4ipdOc6Cm6s
[22] https://engineering.purdue.edu/ChanGroup/ECE595/files/Lecture18_MLP.pdf
[23] https://pabloinsente.github.io/the-multilayer-perceptron
[24] https://www.datacamp.com/blog/machine-learning-projects-for-all-levels
[25] https://www.youtube.com/watch?v=FSiUwFjDTrE
[26] https://stackoverflow.com/questions/tagged/mlp?tab=Votes
[27] https://www.youtube.com/watch?v=pxn_tG2Ddb8
[28] https://www.cs.cmu.edu/~./epxing/Class/10715/lectures/MultiLayerPerceptron.pdf
[29] https://www.mathworks.com/matlabcentral/fileexchange/118105-simple-multi-layer-perceptron-example
[30] https://scte-iitkgp.vlabs.ac.in/exp/multilayer-perceptron/theory.html
[31] https://www.appliedaicourse.com/blog/multilayer-perceptron-in-machine-learning/

---
Answer from Perplexity: pplx.ai/share



# Perceptron and Gradient Descent Questions

## Easy Questions

1. What is the default bias value in the PerceptronNeuron class?
2. How many iterations does the perceptron algorithm perform in the main code?
3. What percentage of data is used for testing in the train_test_split function?
4. What is the purpose of the `predict` method in the PerceptronNeuron class?
5. What activation function is implicitly used in the perceptron implementation?
6. What percentage of labels are flipped when adding noise to the dataset?
7. What is the initial value of weights when training the perceptron?
8. What metric is used to evaluate the performance of the perceptron?
9. What function is used to visualize the decision boundary?
10. What is the shape of the weights vector in the perceptron?

## Medium Questions

1. What is the difference between the `pla_predict` and `predict` methods in the PerceptronNeuron class?
2. How does the perceptron algorithm handle misclassified points during training?
3. What would happen if we increased the number of iterations in the perceptron training?
4. How does the code determine if a point is misclassified in the perceptron algorithm?
5. Why is the bias updated differently than the weights in the perceptron learning algorithm?
6. How does adding noise to the dataset affect the perceptron's performance?
7. What is the purpose of the sigmoid function in the gradient descent implementation?
8. How does the gradient descent learning algorithm differ from the perceptron learning algorithm?
9. What is the role of the learning rate (eta) in the gradient descent algorithm?
10. How is the cross-entropy loss calculated and what does it represent?

## Hard Questions

1. Why might the perceptron algorithm fail to converge, and how would you modify the code to handle this case?
2. How would you implement an early stopping criterion for the perceptron algorithm based on convergence?
3. What modifications would be needed to extend this perceptron to handle multi-class classification?
4. How would you implement regularization in the gradient descent algorithm to prevent overfitting?
5. Explain the mathematical relationship between the perceptron algorithm and the gradient descent approach for logistic regression.
6. How would you modify the code to implement a pocket algorithm to find the best solution when data is not linearly separable?
7. What are the theoretical guarantees of the perceptron algorithm, and under what conditions do they apply?
8. How would you implement a mini-batch version of the gradient descent algorithm?
9. How would you modify the perceptron to implement a margin-based classifier similar to SVM?
10. What changes would be needed to implement a stochastic gradient descent version of the perceptron learning algorithm?

