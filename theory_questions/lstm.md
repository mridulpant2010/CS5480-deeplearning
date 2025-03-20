# LSTM Questions

## ✅ Easy Questions:

1. **What does LSTM stand for?**
   - *(Answer: Long Short-Term Memory)*

2. **What is the purpose of the embedding layer in this LSTM implementation?**
   - *(Answer: Converts input tokens into dense vector representations.)*

3. **How many gates are there in an LSTM cell? Can you name them?**
   - *(Answer: Four gates—input gate, forget gate, output gate, and cell candidate gate.)*

4. **Which activation functions are used in LSTM gates?**
   - *(Answer: Sigmoid (σ) and tanh.)*

5. **What is the role of the forget gate in an LSTM?**
   - *(Answer: Decides what information to discard from the previous cell state.)*

6. **What does the hidden state (`h`) represent in an LSTM?**
   - *(Answer: It represents the LSTM's current memory or context at a given time step.)*

7. **In PyTorch, what does `nn.Parameter` do?**
   - *(Answer: It defines tensors as trainable parameters that will be updated during backpropagation.)*

8. **Why do we initialize hidden and cell states with zeros at the beginning of sequence processing?**
   - *(Answer: To provide a neutral starting point before processing any input data.)*

9. **What is the size/dimension of the hidden state in your provided code?**
   - *(Answer: `[batch_size, hidden_size]`)*

10. **What does the final linear layer (`W_hy` and `b_y`) do in this implementation?**
    - *(Answer: Maps the final hidden state to the desired output dimension.)*

---

## ✅ Medium Questions:

1. **Explain mathematically how the next cell state (`c_next`) is computed in your LSTM implementation.**
   - *(Answer: `c_next = f * c + i * g` where `f` is forget gate output, `i` is input gate output, and `g` is candidate cell state.)*

2. **Why do we use sigmoid activation specifically for gates in LSTMs?**
   - *(Answer: Because sigmoid outputs values between 0 and 1, effectively controlling how much information passes through each gate.)*

3. **How does an LSTM handle vanishing gradient problems better than a standard RNN?**
   - *(Answer: Through gating mechanisms (input, forget, output gates), it selectively retains important information over long sequences, mitigating gradient vanishing issues.)*

4. **What would happen if we remove or disable the forget gate from this implementation?**
   - *(Answer: The model would lose its ability to selectively forget irrelevant past information, potentially causing degraded performance or memory saturation.)*

5. **Can you briefly explain why we have separate weight matrices for input (`W_x*`) and hidden states (`W_h*`) in each gate computation?**
   - *(Answer: To independently control how new inputs and previous hidden states influence each gate's decision-making process.)*

6. **If you had to implement dropout regularization within this LSTM model, at which points would you apply it?**
   - *(Answer: Typically applied after embedding layers or between stacked LSTM layers to prevent overfitting.)*

7. **Why do we use tanh activation for candidate cell state (`g`) instead of sigmoid activation?**
   - *(Answer: Because tanh outputs values between [-1, 1], allowing updates to be both positive or negative, which helps stabilize training.)*

8. **How does batch size impact training efficiency and model performance in this LSTM implementation?**
   - *(Answer: Larger batch sizes can speed up training but may require more memory; smaller batches provide noisier gradient estimates but may generalize better.)*

9. **What happens internally when you call `.to(x.device)` on tensors like hidden states (`h`, `c`)?**
   - *(Answer: Moves tensors to GPU or CPU depending on where input data resides, ensuring compatibility during computations.)*

10. **Can you explain why embeddings are preferred over one-hot encodings for textual data when using RNN/LSTM models?**
    - *(Answer: Embeddings capture semantic relationships between tokens efficiently and reduce dimensionality compared to sparse one-hot encodings.)*

---

## ✅ Hard Questions:

1. **If you wanted to make this LSTM bidirectional, how would you modify your current implementation code-wise and conceptually?**
   - *(Answer: You would add another parallel LSTM processing sequences backward and concatenate or combine forward/backward hidden states at each step or at final output.)*

2. **Explain how gradient clipping might be implemented with your CustomLSTM model during training and why it could be beneficial.**
   - *(Answer: Gradient clipping limits gradients' magnitude during backpropagation preventing exploding gradients; implemented using PyTorch's `torch.nn.utils.clip_grad_norm_()` function after computing gradients.)*

3. **If your sequence length becomes very large (e.g., thousands of time steps), what practical issues might arise with this implementation, and how could you address them?**
   - *(Answer: Issues include memory usage problems and vanishing/exploding gradients; solutions include truncated backpropagation through time (TBPTT), gradient clipping, or attention mechanisms.)*

4. **How would you modify this CustomLSTM implementation to support multiple stacked LSTM layers (multi-layered)? Explain briefly with pseudocode or steps clearly outlined.**
   - *(Answer: You'd instantiate multiple layers of CustomLSTM cells sequentially feeding outputs from one layer as inputs to next; manage hidden/cell states separately per layer.)*

5. **Explain clearly how backpropagation through time (BPTT) works specifically within this CustomLSTM implementation during training.**
   - *(Answer: BPTT unrolls sequence computations backward through each time step computing gradients recursively from final outputs back through all gates/cell updates at each time step.)*

6. **Can you describe a scenario where using an LSTM like this might perform worse than simpler models like CNNs or feed-forward networks? Why would that happen?**
   - *(Answer: For short sequences without temporal dependencies or when parallelization matters significantly; CNNs/feed-forward networks may outperform due to simplicity/speed advantages.)*

7. **If you observe that training loss isn't decreasing after several epochs with this CustomLSTM model, what debugging steps would you take systematically to identify potential issues?**
   - *(Answer: Steps include checking data preprocessing correctness, verifying model parameter initialization methods, ensuring correct loss computation/optimizer setup, checking gradient flow via `.grad` attributes/gradient norms.)*

8. **Explain how weight initialization affects training convergence specifically for gates in your CustomLSTM implementation—what initialization methods are recommended for these parameters typically?**
   - *(Answer: Poor initialization can cause unstable gradients; recommended methods include Xavier/Glorot initialization or orthogonal initialization for recurrent weights.)*

9. **If you wanted to implement attention mechanisms on top of your CustomLSTM outputs, briefly outline how you'd integrate attention into your current forward pass logic clearly mentioning required changes/additions explicitly.**
    - *(Answer: Compute attention scores using hidden states across sequence positions; apply softmax normalization; compute weighted sum of hidden states based on attention weights before final linear output layer computation.)*

10. **How would you adapt your CustomLSTM model if inputs were continuous numerical vectors instead of discrete token indices requiring embeddings—what specific changes would be necessary within your existing forward method/code structure clearly stated step-by-step explicitly here?**
    - *(Answer: Remove embedding layer entirely; directly feed numerical input vectors into gate computations replacing embedded inputs directly with continuous numerical vectors at each timestep computation inside forward loop.)* 

---

These structured questions progressively test basic understanding (easy), deeper conceptual comprehension (medium), and practical problem-solving/application skills (hard), providing comprehensive preparation for viva sessions related specifically to your CustomLSTM implementation code provided above!

---
