import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, epochs=100, dropout=0.3):
        super(FFNN, self).__init__()
        self.lr = lr
        self.epochs = epochs

        # Fixed: fc2 input was output_dim, should be hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Used to map integer preds back to string labels
        self.classes = None
        self.class_to_idx = None

    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.dropout(self.relu(z1))
        z2 = self.fc2(a1)
        return z2  # raw logits — CrossEntropyLoss handles softmax internally

    def fit(self, X, y):
        # Encode string labels to integers
        self.classes = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_encoded = np.array([self.class_to_idx[c] for c in y])

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y_encoded)

        # Class weights to handle CD4/CD8 imbalance
        class_counts = np.bincount(y_encoded)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(self.classes)
        weights_t = torch.FloatTensor(class_weights)

        criterion = nn.CrossEntropyLoss(weight=weights_t)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Adam > SGD for this data

        self.train()
        self.loss_history = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.forward(X_t)
            loss = criterion(logits, y_t)
            loss.backward()  # PyTorch computes all gradients automatically
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{self.epochs} | Loss: {loss.item():.4f}")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            logits = self.forward(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
            pred_indices = np.argmax(probs, axis=1)

        # Map back to string labels to match kNN and RF output format
        pred_labels = np.array([self.classes[i] for i in pred_indices])
        return pred_labels, probs