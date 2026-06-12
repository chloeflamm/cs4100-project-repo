"""
ffnn.py
Author: Nasir Stanley

Feed Forward Neural Network (FFNN) implemented in PyTorch for T-cell activation state classification from PCA-reduced scRNA-seq data.

Architecture:
Input(50 components) -> Linear -> ReLU -> Dropout -> Linear -> Softmax
Trained with Adam optimizer and class-weighted cross-entropy loss to handle CD4/CD8 class imbalance in the dataset.

AI Usage: How can I turn my built out numpy FFNN into a PyTorch acceptable network?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, epochs=100, dropout=0.3):
        super(FFNN, self).__init__()
        self.lr = lr
        self.epochs = epochs

        # fc1: input layer -> hidden layer (learns initial feature representations)
        # fc2: hidden layer -> output layer (maps to class scores)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Small random weights to prevent symmetry breaking at initialization
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

        self.relu = nn.ReLU() # ReLU introduces non-linearity

        # Dropout randomly zeros neurons during training to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Stored during fit() to convert integer predictions back to cell type strings
        self.classes = None
        self.class_to_idx = None

    def forward(self, X):
        z1 = self.fc1(X) # linear transformation: X @ W1 + b1
        a1 = self.dropout(self.relu(z1))  # non-linear activation + regularization
        z2 = self.fc2(a1) # linear transformation: a1 @ W2 + b2
        return z2  # raw logits

    def fit(self, X, y):
        # store mapping so predict() can convert back to strings
        self.classes = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_encoded = np.array([self.class_to_idx[c] for c in y])

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y_encoded)

        # Since CD4 cells outnumber CD8 the class weights penalize the model more for misclassifying
        # minority classes so it doesn't just predict CD4 for everything
        class_counts = np.bincount(y_encoded)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(self.classes)
        weights_t = torch.FloatTensor(class_weights)

        # CrossEntropyLoss combines log softmax + negative log likelihood
        criterion = nn.CrossEntropyLoss(weight=weights_t)

        # Adam adapts learning rate per parameter that converges faster than SGD on high dimensional sparse data like gene expression
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.train()  # enables dropout during training
        self.loss_history = []

        for epoch in range(self.epochs):
            optimizer.zero_grad() # clear gradients from previous step
            logits = self.forward(X_t) # forward pass
            loss = criterion(logits, y_t) # compute loss
            loss.backward() # backprop
            optimizer.step() # update weights

            self.loss_history.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} | Loss: {loss.item():.4f}")

    def predict(self, X):
        self.eval()  # disables dropout for inference
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits = self.forward(X_t)
            # Converting logits to probabilities
            probs = torch.softmax(logits, dim=1).numpy()
            pred_indices = np.argmax(probs, axis=1)

        # Convert integer indices back to cell type strings
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        return pred_labels, probs