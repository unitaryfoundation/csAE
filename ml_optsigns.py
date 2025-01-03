import numpy as np
import matplotlib.pyplot as plt
from signals import *
import torch
from sklearn.model_selection import train_test_split
from torch import nn

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / (y_pred.shape[0]*y_pred.shape[1])) * 100
    return acc
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model
class SignModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),  # how many classes are there?
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

class SignModelDeep(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units//4),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units//4, out_features=hidden_units//2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units // 2, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),  # how many classes are there?
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

if __name__ == "__main__":

    # For reproducibility
    np.random.seed(14)
    # Set the per oracle noise parameter (See Eq. 18)
    eta = 0
    # Set array parameters
    narray = [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    narray = [2]*12
    # Set sampling rate
    C = 3.0

    # generate training data
    # num_train = 500000 #93.97%
    # epochs = 2000
    # num_train = 100000 #94.81%
    # epochs = 1000
    # num_train = 500000 #95.22%
    # epochs = 2000
    # num_train = 50000 #95.55 with extra layers
    # epochs = 5000

    num_train = 50000
    epochs = 10

    # Use for length
    # This sets up the simulation that simulates the measured amplitudes at the various physical locations.
    # It uses a C=1.5 value, which corresponds to the sampling schedule given in Eq. 16. The variable C here
    # is the parameter K in the paper.
    ula_signal = TwoqULASignal(M=narray, C=C)
    depths, _ = ula_signal._get_depths(narray, C=C)
    print(depths)
    NUM_FEATURES = len(depths)
    NUM_CLASSES = len(depths)-1
    measurements = np.zeros((num_train, NUM_FEATURES), dtype=np.float64)
    exact_signs = np.zeros((num_train, NUM_CLASSES), dtype=np.int32)

    # Create an instance of BlobModel and send it to the target device
    sign_model = SignModel(input_features=NUM_FEATURES,
                        output_features=NUM_CLASSES,
                        hidden_units=16*NUM_FEATURES).to(device)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(sign_model.parameters(), lr=0.15, momentum=0.9) #93.97%
    optimizer = torch.optim.SGD(sign_model.parameters(), lr=0.15, momentum=0.98)  # 94.81%
    optimizer = torch.optim.SGD(sign_model.parameters(), lr=0.15, momentum=0.98)  # 95.12% shallow model
    # optimizer = torch.optim.SGD(sign_model.parameters(), lr=0.1, momentum=0.98)

    for i in range(num_train):
        theta = np.random.uniform(0, np.pi/2.0)
        signal = ula_signal.estimate_signal(n_samples=ula_signal.n_samples, theta=theta, eta=eta)
        measurements[i,:] = ula_signal.measurements[:]
        exact_signs[i,:] = 0.5*(1+np.array(ula_signal.signs_exact[1:])) # make signs 0 or 1

    X = torch.from_numpy(measurements).type(torch.float)
    y = torch.from_numpy(exact_signs).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2, # 20% test, 80% train
                                                        random_state=42) # make the random split reproducible

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # batch_size = num_train // epochs
    batch_size = 500
    batch_start = torch.arange(0, len(X_train), batch_size)

    for epoch in range(epochs):
        for start in batch_start:
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            ### Training
            sign_model.train()

            # 1. Forward pass
            y_logits = sign_model(X_train) # model outputs raw logits
            y_pred = (torch.sigmoid(y_logits) > 0.5).float()

            # 2. Calculate loss and accuracy
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_true=y_train,
                              y_pred=y_pred)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backwards
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        ### Testing
        sign_model.eval()
        with torch.inference_mode():
          # 1. Forward pass
          test_logits = sign_model(X_test)
          test_pred = (torch.sigmoid(test_logits) > 0.5).float()
          # 2. Calculate test loss and accuracy
          test_loss = loss_fn(test_logits, y_test)
          test_acc = accuracy_fn(y_true=y_test,
                                 y_pred=test_pred)

        # Print out what's happening
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

    print(test_logits[1:2], y_test[1:2])
    print(test_pred[1:2], y_test[1:2])

    file_subscript = ''
    for x in narray:
        file_subscript += f'{x}'

    filename = f'ml_models/sign_model_{file_subscript}_C{C:0.2f}.pt'
    torch.save(sign_model.state_dict(), filename, pickle_protocol=4)
    # torch.save(sign_model, "ml_models/sign_model_dropout_less_222223.pt")
    # print(measurements)
    # print(exact_signs)
