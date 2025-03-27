import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import random

def subsample_words(int_words, threshold=1e-5):
    """
    Subsampling of frequent words: removes words that appear with higher frequency
    than threshold with probability proportional to their frequency.
    
    Args:
        int_words: List of word indices
        threshold: Threshold for subsampling (default: 1e-5)
    
    Returns:
        List of subsampled word indices
    """
    word_counts = Counter(int_words)
    total_n_words = len(int_words)

    # Calculate frequency ratios
    freq_ratios = {word: count/total_n_words for word, count in word_counts.items()}
    
    # Calculate probability of dropping each word
    p_drop = {word: 1 - np.sqrt(threshold/freq_ratios[word]) for word in word_counts if freq_ratios[word] > threshold}
    
    # Keep words with probability 1 - p_drop
    subsampled_words = [word for word in int_words if word not in p_drop or random.random() > p_drop[word]]
    
    return subsampled_words

class SkipGram:
    def __init__(self, input, hidden, output=1, neg_sample_size=5):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.neg_sample_size = neg_sample_size  # Number of negative samples
        
        # Initialize weights and biases
        self.W1 = torch.tensor(np.random.randn(input, hidden), dtype=torch.float32)  # Input × Hidden
        self.W2 = torch.tensor(np.random.randn(hidden, output), dtype=torch.float32)  # Hidden × Output
        self.b1 = torch.tensor(np.random.randn(hidden), dtype=torch.float32)  # Hidden bias
        self.b2 = torch.tensor(np.random.randn(output), dtype=torch.float32)  # Output bias
        
        self.eta = 0.01  # Learning rate

    def forward(self, input_tensor):
        # Forward pass
        f = torch.tanh(torch.matmul(input_tensor, self.W1) + self.b1)  # Hidden layer activation
        return f  # Return only hidden activations
    
    def sample_negatives(self, vocab_size, exclude_indices, n_samples):
        """
        Negative sampling: Draw 'n_samples' random indices, excluding 'exclude_indices'.
        """
        neg_indices = []
        while len(neg_indices) < n_samples:
            sample = np.random.randint(0, vocab_size)
            if sample not in exclude_indices:  # Avoid sampling positive targets
                neg_indices.append(sample)
        return torch.tensor(neg_indices, dtype=torch.long)

    def calculate_loss(self, f, target_word_idx, neg_sample_indices, W2, b2):
        """
        Compute the binary cross-entropy loss for positive and negative samples.
        """
        # Positive sample score
        pos_score = torch.sigmoid(torch.matmul(f, W2[:, target_word_idx]) + b2[target_word_idx])
        pos_loss = -torch.log(pos_score + 1e-9)  # Add small epsilon for numerical stability

        # Negative samples score
        neg_scores = torch.sigmoid(torch.matmul(f, W2[:, neg_sample_indices]) + b2[neg_sample_indices])
        neg_loss = -torch.sum(torch.log(1 - neg_scores + 1e-9))  # Sum over negative samples

        return pos_loss + neg_loss

    def gradient_descent(self, input_tensor, target_indices, vocab_size):
        """
        Perform gradient descent using negative sampling.
        """
        n_epoch = 10
        for epoch in range(n_epoch):
            total_loss = 0.0
            for i, input_vec in enumerate(input_tensor):
                # Forward pass
                f = self.forward(input_vec.unsqueeze(0))

                # Positive target index
                target_idx = target_indices[i]

                # Negative sampling
                neg_sample_indices = self.sample_negatives(vocab_size, [target_idx], self.neg_sample_size)

                # Loss computation
                loss = self.calculate_loss(f, target_idx, neg_sample_indices, self.W2, self.b2)
                total_loss += loss.item()

                # Gradients for W2 and b2
                pos_score = torch.sigmoid(torch.matmul(f, self.W2[:, target_idx]) + self.b2[target_idx])
                self.W2[:, target_idx] -= self.eta * (pos_score - 1) * f.squeeze()
                self.b2[target_idx] -= self.eta * (pos_score - 1)

                neg_scores = torch.sigmoid(torch.matmul(f, self.W2[:, neg_sample_indices]) + self.b2[neg_sample_indices])
                for j, neg_idx in enumerate(neg_sample_indices):
                    self.W2[:, neg_idx] -= self.eta * neg_scores[j] * f.squeeze()
                    self.b2[neg_idx] -= self.eta * neg_scores[j]

                # Gradients for W1 and b1
                grad_f = (pos_score - 1) * self.W2[:, target_idx] + torch.sum(neg_scores.unsqueeze(1) * self.W2[:, neg_sample_indices], dim=1)
                self.W1 -= self.eta * torch.matmul(input_vec.unsqueeze(1), grad_f.unsqueeze(0))
                self.b1 -= self.eta * grad_f.squeeze()

            print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, text, window_size=2, num_negative_samples=5, subsample_threshold=1e-5):
        words = text.lower().split()
        word_counts = Counter(words)
        self.vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        # Convert text to word indices
        word_indices = [self.vocab[word] for word in words if word in self.vocab]
        
        # Apply subsampling to word indices
        subsampled_indices = subsample_words(word_indices, threshold=subsample_threshold)
        print(f"Subsampling reduced word count from {len(word_indices)} to {len(subsampled_indices)}")
        
        # Generate skip-gram pairs
        self.skip_grams = []
        for i in range(len(subsampled_indices)):
            for j in range(max(0, i - window_size), min(len(subsampled_indices), i + window_size + 1)):
                if i != j:
                    self.skip_grams.append((subsampled_indices[i], subsampled_indices[j]))
        
        self.num_negative_samples = num_negative_samples
        
        # Create sampling distribution for negative sampling (unigram^0.75)
        word_freqs = np.array([count for _, count in word_counts.most_common()])
        word_freqs = word_freqs ** 0.75
        self.sampling_weights = word_freqs / np.sum(word_freqs)

    def __len__(self):
        return len(self.skip_grams)

    def __getitem__(self, idx):
        target_word, context_word = self.skip_grams[idx]
        
        # Generate negative samples
        negative_samples = []
        while len(negative_samples) < self.num_negative_samples:
            neg_idx = np.random.choice(self.vocab_size, p=self.sampling_weights)
            if neg_idx != context_word and neg_idx not in negative_samples:
                negative_samples.append(neg_idx)
                
        return torch.tensor(target_word), torch.tensor(context_word), torch.tensor(negative_samples)

def train_skip_gram_model(text, embedding_dim=100, window_size=2, 
                         num_negative_samples=5, subsample_threshold=1e-5,
                         batch_size=32, epochs=10):
    """
    Train a Skip-Gram model with negative sampling and subsampling
    
    Args:
        text: Input text corpus
        embedding_dim: Dimension of word embeddings
        window_size: Context window size
        num_negative_samples: Number of negative samples per positive sample
        subsample_threshold: Threshold for subsampling frequent words
        batch_size: Batch size for training
        epochs: Number of training epochs
    
    Returns:
        Trained model, vocabulary, and index-to-word mapping
    """
    # Create dataset with subsampling
    dataset = SkipGramDataset(
        text, 
        window_size=window_size, 
        num_negative_samples=num_negative_samples,
        subsample_threshold=subsample_threshold
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Initialize model
    model = SkipGram(
        input=dataset.vocab_size,
        hidden=embedding_dim,
        output=dataset.vocab_size,
        neg_sample_size=num_negative_samples
    )
    
    # Prepare input tensors and target indices for training
    input_tensors = []
    target_indices = []
    
    for target_word, context_word, _ in dataloader:
        # Create one-hot encoded input vectors
        for target in target_word:
            input_vec = torch.zeros(dataset.vocab_size)
            input_vec[target.item()] = 1
            input_tensors.append(input_vec)
        
        # Add context words as targets
        for context in context_word:
            target_indices.append(context.item())
    
    # Train the model
    model.gradient_descent(
        torch.stack(input_tensors),
        target_indices,
        dataset.vocab_size
    )
    
    return model, dataset.vocab, dataset.idx_to_word

def find_similar_words(word, model, vocab, idx_to_word, top_k=5):
    """Find words most similar to the input word"""
    if word not in vocab:
        return []
    
    word_idx = vocab[word]
    
    # Create one-hot encoded vector for the word
    input_vec = torch.zeros(len(vocab))
    input_vec[word_idx] = 1
    
    # Get the word embedding (hidden layer activation)
    word_embedding = model.forward(input_vec.unsqueeze(0)).squeeze().detach().numpy()
    
    # Calculate similarity with all other words
    similarities = []
    for idx in range(len(vocab)):
        if idx != word_idx:
            # Create one-hot encoded vector for comparison word
            comp_vec = torch.zeros(len(vocab))
            comp_vec[idx] = 1
            comp_embedding = model.forward(comp_vec.unsqueeze(0)).squeeze().detach().numpy()
            
            # Calculate cosine similarity
            similarity = np.dot(word_embedding, comp_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(comp_embedding)
            )
            similarities.append((idx_to_word[idx], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example usage
if __name__ == "__main__":
    text = """
    The quick brown fox jumps over the lazy dog. The dog was not very lazy, but rather tired.
    Foxes are known for their quickness and agility. Dogs are known for their loyalty and friendship.
    The quick brown fox and the lazy dog are often used in typing exercises.
    """
    
    model, vocab, idx_to_word = train_skip_gram_model(
        text,
        embedding_dim=50,
        window_size=2,
        num_negative_samples=5,
        subsample_threshold=1e-5,
        epochs=10
    )
    
    # Find similar words
    for word in ["the", "fox", "dog"]:
        similar_words = find_similar_words(word, model, vocab, idx_to_word)
        print(f"Words similar to '{word}': {similar_words}")
