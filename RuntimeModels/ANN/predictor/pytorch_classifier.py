import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

# Nested ANNModel class inside PyTorchClassifier
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        in_features = input_dim
        # Add hidden layers with ReLU activations and dropout
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        # Output layer without sigmoid since BCEWithLogitsLoss applies sigmoid internally
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Wrapper class to integrate PyTorch model with scikit-learn's API
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_layers=None, dropout_rate=None, learning_rate=None, num_epochs=None, batch_size=None, patience=None, accumulation_steps=None):
        # Initialize the model with parameters provided by RandomizedSearchCV or default to None
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model if parameters are provided, else set to None (will be set during fit)
        if all(param is not None for param in [hidden_layers, dropout_rate, learning_rate, num_epochs, batch_size, patience, accumulation_steps]):
            self._initialize_model()
        else:
            self.model = None
    
    def _initialize_model(self):
        # This function initializes the model, optimizer, and other components
        self.model = self.ANNModel(input_dim=self.input_dim, hidden_layers=self.hidden_layers, dropout_rate=self.dropout_rate).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)  # Use BCEWithLogitsLoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = GradScaler()
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        # If the model hasn't been initialized, initialize it now with the given parameters
        if self.model is None:
            self._initialize_model()

        self.classes_ = np.unique(y)

        # Lists to store metrics for analysis
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_losses = []
        self.val_losses = []

        # Convert input data to NumPy arrays if they are pandas DataFrames or Series
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Convert the data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Create DataLoader for batching the training data
        train_data = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        best_loss = np.inf  # Initialize best loss for early stopping
        patience_counter = 0  # Counter for early stopping patience

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            y_train_pred = []
            y_train_true = []

            # Gradient accumulation across batches
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()  # Zero the gradients
                with autocast():
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, labels) / self.accumulation_steps
                self.scaler.scale(loss).backward()  # Scale the loss and backpropagate

                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)  # Update the model parameters
                    self.scaler.update()  # Update the scaler for mixed precision

                epoch_loss += loss.item() * self.accumulation_steps

                y_train_pred.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int))  # Apply sigmoid for predictions
                y_train_true.extend(labels.cpu().numpy().astype(int))

            # Calculate training F1 score for the epoch
            train_f1 = f1_score(y_train_true, y_train_pred)
            self.train_f1_scores.append(train_f1)
            self.train_losses.append(epoch_loss / len(train_loader))

            # Validation phase
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                    y_val_pred = (torch.sigmoid(val_outputs) > 0.5).cpu().numpy().astype(int)  # Apply sigmoid for predictions
                    val_f1 = f1_score(y_val_tensor.cpu().numpy().astype(int), y_val_pred)

                self.val_f1_scores.append(val_f1)
                self.val_losses.append(val_loss)

                # Print epoch progress with validation metrics
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

                # Clear GPU cache after each epoch
                del inputs, labels, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0  # Reset patience counter if the model improves
                else:
                    patience_counter += 1  # Increment patience counter if no improvement

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}")

        return self

    # Prediction method
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)  # Apply sigmoid for predictions
        return predictions

    # Predict probabilities for binary classification
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            probabilities = np.stack([(1 - probabilities), probabilities], axis=1)  # Format for binary classification
        return probabilities