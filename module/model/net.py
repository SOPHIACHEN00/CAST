import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Union
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score
import time

# NEW!
from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# 


# https://blog.csdn.net/sxf1061700625/article/details/105870851?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166763065216800180665746%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166763065216800180665746&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105870851-null-null.142^v63^pc_rank_34_queryrelevant25,201^v3^control_2,213^v1^control&utm_term=mnist&spm=1018.2226.3001.4187

# NEW!
class Net(nn.Module):
    def __init__(self, seed, lr=None, num_epoch=None, device=None, batch_size=None):
        self.seed = seed
        self.lr = lr
        self.num_epoch = num_epoch
        self.device = device
        self.batch_size = batch_size
        self.incremental_seed = seed
        torch.manual_seed(self.seed)
        super().__init__()

    def _score_accuracy_gpu(self, X_test, y_test, batch_size):
        torch.manual_seed(self.seed)
        self.eval()

        if torch.is_tensor(y_test):
            y_true = y_test.to(self.device)
        else:
            y_true = torch.as_tensor(y_test, device=self.device)

        correct = torch.zeros((), device=self.device, dtype=torch.float32)
        total = torch.zeros((), device=self.device, dtype=torch.float32)

        test_dataset = torch.utils.data.TensorDataset(X_test, y_true.detach().cpu())
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for xb, yb in test_dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                outputs = self(xb)

                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    preds = torch.argmax(outputs, dim=1)
                    targets = yb.view(-1).long()
                else:
                    preds = (outputs.view(-1) >= 0.5).long()
                    targets = yb.view(-1).long()

                correct += (preds == targets).float().sum()
                total += float(targets.numel())

        return float((correct / total.clamp_min(1.0)).item())

# class Net(nn.Module):
#     def __init__(self, seed):
#         self.seed = seed
#         self.incremental_seed = seed
#         torch.manual_seed(self.seed)
#         super().__init__()
#         return

    def _fed_train(self,
                   X_train_parts,
                   y_train_parts,
                   num_global_rounds,
                   num_local_rounds,
                   loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
                   lr,
                   score: bool,
                   batch_size,
                   **kwargs
                   ):

        self.load_state_dict(self.initial_state_dict)
        self.incremental_seed = self.seed
        torch.manual_seed(self.seed)

        if score:
            test_interval = kwargs.get("test_interval")
            value_functions = kwargs.get("value_functions")
            X_test, y_test = kwargs.get("X_test"), kwargs.get("y_test")
            val_list = np.zeros(len(value_functions))

        num_clients = len(X_train_parts)

        self.load_state_dict(self.initial_state_dict)
        device = self.device

        m = np.zeros(num_clients)
        for i in range(num_clients):
            m[i] = X_train_parts[i].size(0)

        # start training
        for t in range(num_global_rounds):
            # distribute model, make sure the Adam is initialized locally
            backup_model = copy.deepcopy(self)
            # it is the same as the following:
            # backup_model = self.__class__(seed=self.seed, lr=self.lr, num_epoch=self.num_epoch, hidden_layer_size=self.hidden_layer_size, device=self.device, batch_size=self.batch_size)
            # backup_model.load_state_dict(self.state_dict())
            models = [copy.deepcopy(backup_model) for _ in range(num_clients)]
            deltas = []

            # -------  client update  -----------
            for i in range(num_clients):
                # models[i] = copy.deepcopy(backup_model)
                models[i] = self.client_update(X_train_parts[i], y_train_parts[i], models[i], num_local_rounds)
                deltas.append(self.compute_grad_update(old_model=backup_model, new_model=models[i], device=device))

            # -------- run on server side ---------
            # FedAvg
            weights = m / np.sum(m)
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.parameters()]
            for delta, weight in zip(deltas, weights):
                self.add_gradient_updates(aggregated_gradient, delta, weight=weight)
            self = self.add_update_to_model(self, aggregated_gradient)

            if score:
                if t % test_interval == 0 and t != 0:
                    y_pred = self._predict(X_test, batch_size)
                    for idx_val, temp_value_function in enumerate(value_functions):
                        if temp_value_function == "accuracy":
                            val_list[idx_val] = max(val_list[idx_val], accuracy_score(y_true=y_test, y_pred=y_pred))
                        elif temp_value_function == "f1":
                            val_list[idx_val] = max(val_list[idx_val], f1_score(y_true=y_test, y_pred=y_pred))
                        elif temp_value_function == "f1_macro":
                            val_list[idx_val] = max(val_list[idx_val],
                                                    f1_score(y_true=y_test, y_pred=y_pred, average="macro"))
                        elif temp_value_function == "f1_micro":
                            val_list[idx_val] = max(val_list[idx_val],
                                                    f1_score(y_true=y_test, y_pred=y_pred, average="micro"))
        return val_list

    @staticmethod
    def client_update(X_train_client, y_train_client, model_last_round, num_local_epochs):
        new_model = copy.deepcopy(model_last_round)
        new_model.fit(X_train_client, y_train_client, incremental=True, num_epochs=num_local_epochs)
        return new_model

    @staticmethod
    def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
        assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"

        for param_1, param_2 in zip(grad_update_1, grad_update_2):
            param_1.data += param_2.data * weight

    @staticmethod
    def add_update_to_model(model, update, weight=1.0, device=None):
        if not update:
            return model
        if device:
            model = model.to(device)
            update = [param.to(device) for param in update]

        for param_model, param_update in zip(model.parameters(), update):
            param_model.data += weight * param_update.data
        return model

    @staticmethod
    def compute_grad_update(old_model, new_model, device=None):
        # maybe later to implement on selected layers/parameters
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        return [(new_param.data - old_param.data) for old_param, new_param in
                zip(old_model.parameters(), new_model.parameters())]

    @staticmethod
    def flatten(grad_update):
        return torch.cat([update.data.view(-1) for update in grad_update])

    @staticmethod
    def unflatten(flattened, normal_shape):
        grad_update = []
        for param in normal_shape:
            n_params = len(param.view(-1))
            grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
            flattened = flattened[n_params:]
        return grad_update

    # Make sure to initialize once before each fit
    def _fit(self,
             X_train,
             y_train,
             num_epochs,
             loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
             lr,
             incremental: bool,
             batch_size
             ):

        # Initialize once to prevent the last training from affecting this one
        if not incremental:
            self.load_state_dict(self.initial_state_dict)
            self.incremental_seed = self.seed
            torch.manual_seed(self.seed)
        else:
            self.incremental_seed += 17
            torch.manual_seed(self.incremental_seed)

        loss_fun = loss_fun.to(self.device)
        # print(f"lenX = {len(X_train)}")
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # first, process data. put into dataloader.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # enter training mode
        self.train(True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # print(optimizer.state_dict())
        # optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if isinstance(loss_fun, nn.CrossEntropyLoss):
                    labels = labels.long().view(-1)
                else:
                    labels = labels.unsqueeze(1).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self(inputs)

                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Optional instrumentation hook used by WEF collection. It is
            # intentionally attribute-based so existing fit APIs and all
            # non-WEF training paths remain unchanged.
            epoch_end_callback = getattr(self, "_epoch_end_callback", None)
            if epoch_end_callback is not None:
                epoch_end_callback(self, epoch)
        #exit training mode
        self.train(False)
        return

    def _fit_and_score(self, X_train, y_train, X_test, y_test, value_functions, num_epochs,
                       # incremental=False,
                       loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
                       lr,
                       test_interval,
                       batch_size
                       ):
        torch.manual_seed(self.seed)

        loss_fun = loss_fun.to(self.device)

        # print(f"lenX = {len(X_train)}")
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # first, process data. put into dataloader.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize once to prevent the last training from affecting this one
        self.load_state_dict(self.initial_state_dict)

        # enter training mode
        optimizer = optim.Adam(self.parameters(), lr=lr)

        val_list = np.zeros(len(value_functions))

        for epoch in range(num_epochs):
            self.train(True)
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if isinstance(loss_fun, nn.CrossEntropyLoss):
                    labels = labels.long().view(-1)
                else:
                    labels = labels.unsqueeze(1).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self(inputs)

                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # validate
            if epoch % test_interval == 0 and epoch != 0:
                y_pred = self._predict(X_test, batch_size)
                for idx_val, temp_value_function in enumerate(value_functions):
                    if temp_value_function == "accuracy":
                        val_list[idx_val] = max(val_list[idx_val], accuracy_score(y_true=y_test, y_pred=y_pred))
                    elif temp_value_function == "f1":
                        val_list[idx_val] = max(val_list[idx_val], f1_score(y_true=y_test, y_pred=y_pred))
                    elif temp_value_function == "f1_macro":
                        val_list[idx_val] = max(val_list[idx_val],
                                                f1_score(y_true=y_test, y_pred=y_pred, average="macro"))
                    elif temp_value_function == "f1_micro":
                        val_list[idx_val] = max(val_list[idx_val],
                                                f1_score(y_true=y_test, y_pred=y_pred, average="micro"))

            # print(running_loss)
        # exit training mode
        self.train(False)
        return val_list

    def _predict(self,
                 X_test: torch.tensor,
                 batch_size,
                 ):
        torch.manual_seed(self.seed)
        self.eval()
        predicted_labels = []

        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Iterate over the test dataset
        for data in test_dataloader:
            # print(type(data))
            (inputs,) = data
            inputs = inputs.to(self.device)

            # Forward pass through the model
            outputs = self(inputs)
            outputs = outputs.to("cpu")

            # Get the predicted labels
            predicted = (outputs >= 0.5).squeeze().long()
            predicted_labels.extend(predicted.tolist())

        # Convert the lists to NumPy arrays
        predicted_labels = np.array(predicted_labels)

        return predicted_labels


# class CNNMNIST(Net):
#     def __init__(self, seed, device):
#         super().__init__(seed=seed, device=device)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(7 * 7 * 32, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#         self.initial_state_dict = copy.deepcopy(self.state_dict())
#         return
#
#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 7 * 7 * 32)
#         x = self.relu3(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#     # Suppose training for two epochs?
#     def fit(self, X_train, y_train):
#         return self._fit(X_train, y_train, num_epochs=10, loss_fun=nn.CrossEntropyLoss(), lr=0.01)
#
#     def predict(self, X_test):
#         return self._predict(X_test)
#
#     def _predict(self,
#                  X_test: torch.tensor,
#                  # loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss()] = nn.BCEWithLogitsLoss(),
#                  # test_losses=None
#                  ):
#         torch.manual_seed(self.seed)
#         self.eval()
#         predicted_labels = []
#
#         test_dataset = torch.utils.data.TensorDataset(X_test)
#         test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#         # Iterate over the test dataset
#         for data in test_dataloader:
#             inputs = data
#
#             # Forward pass through the model
#             outputs = self(inputs)
#
#             # Get the predicted labels
#             # _, predicted = torch.max(outputs.data, 1)
#             _, predicted = torch.max(outputs.data, 1)
#             predicted_labels.extend(predicted.tolist())
#
#         # Convert the lists to NumPy arrays
#         predicted_labels = np.array(predicted_labels)
#
#         return predicted_labels

class MNISTMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(MNISTMLP, self).__init__(seed)
        self.name = "MNISTMLP"

        input_size = 28 * 28
        output_size = 10  # output logits (not sigmoid)

        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)

        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if isinstance(x, (list, tuple)):
            x = x[0]

        # flatten to (N, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)

        out = self.fc1(x)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits  # return logits and give i to CrossEntropyLoss

    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(
            X_train, y_train,
            num_epochs=num_epochs,
            loss_fun=nn.CrossEntropyLoss(),
            lr=self.lr,
            incremental=incremental,
            batch_size=self.batch_size
        )

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(
            X_train_parts, y_train_parts,
            num_global_rounds=self.num_epoch,
            num_local_rounds=1,
            loss_fun=nn.CrossEntropyLoss(),
            lr=self.lr,
            test_interval=1,
            batch_size=self.batch_size,
            score=True,
            X_test=X_test, y_test=y_test,
            value_functions=value_functions
        )

    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]

    def predict(self, X_test):
        torch.manual_seed(self.seed)
        self.eval()

        preds = []
        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)

                if x.dim() == 4:
                    x = x.view(x.size(0), -1)
                elif x.dim() == 3:
                    x = x.view(x.size(0), -1)

                logits = self(x)
                predicted = torch.argmax(logits, dim=1)
                preds.extend(predicted.detach().cpu().tolist())

        return np.array(preds, dtype=np.int64)



class IMDBMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size, input_size):
        super(IMDBMLP, self).__init__(seed)
        self.name = "IMDBMLP"
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.input_size = int(input_size)
        self.fc1 = nn.Linear(self.input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]

    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                               loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size,
                               score=True, X_test=X_test, y_test=y_test, value_functions=value_functions)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class AdultMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(AdultMLP, self).__init__(seed)
        self.name = "AdultMLP"
        input_size = 105
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    # NEW!
    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]

    # def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
    #     return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
    #                                loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)
    # num epoch:40, lr: 0.001, accu:0.854793793926535

    # updated 2024-3-25 06:14:05
    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class BankMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(BankMLP, self).__init__(seed)
        self.name = "BankMLP"
        input_size = 51
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    # NEW!
    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]


    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

    # num epoch:40, lr: 0.001, accu:0.9082544457223746
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class TicTacToeMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(TicTacToeMLP, self).__init__(seed)
        self.name = "TicTacToeMLP"
        input_size = 27
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    # NEW!
    def score(self, X_test, y_test, value_functions):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]


    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class Dota2MLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(Dota2MLP, self).__init__(seed)
        self.name = "Dota2MLP"
        input_size = 172
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
    # NEW!
    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class CreditCardMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(CreditCardMLP, self).__init__(seed)
        self.name = "CreditCardMLP"
        input_size = 29
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)



# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# early stopper is only based on the trained train_Change whether the loss changes or not
class EarlyStopper:
    def __init__(self, n_iter_no_change=10, tol=1e-4):
        self.best_model_state = None
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self._no_improvement_count = 0
        self.best_loss = np.inf

    def early_stop(self, train_loss, model):
        # print(f"best loss is {self.best_loss}")
        # print(f"train loss is {train_loss}")
        if train_loss > self.best_loss - self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self.best_model_state = copy.deepcopy(model.state_dict())

        if self._no_improvement_count >= self.n_iter_no_change:
            return True, self.best_model_state
        else:
            return False, None



class AdultLR(AdultMLP):
    def __init__(self, seed, lr, num_epoch, device, batch_size, hidden_layer_size=1):
        super(AdultLR, self).__init__(seed, lr, num_epoch, hidden_layer_size, device, batch_size)
        self.name = "AdultLR"
        self.fc1 = nn.Linear(105, 1)
        self.fc2 = None
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        self.to(device)
        self.device = device

    def forward(self, x):
        out = self.fc1(x)
        return self.sigmoid(out)
    
    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)

class Dota2LR(Net):
    def __init__(self, seed, lr, num_epoch, device, batch_size):
        super().__init__(seed, lr, num_epoch, device, batch_size)
        self.linear = nn.Linear(172, 1)
        self.to(self.device)

class Dota2LR(Dota2MLP):
    def __init__(self, seed, lr, num_epoch, device, batch_size, hidden_layer_size=1):
        super(Dota2LR, self).__init__(seed, lr, num_epoch, hidden_layer_size, device, batch_size)
        self.name = "Dota2LR"
        self.fc1 = nn.Linear(172, 1)
        self.fc2 = None
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        self.to(device)
        self.device = device


    def forward(self, x):
        out = self.fc1(x)
        return self.sigmoid(out)
    
    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class TicTacToeLR(Net):
    def __init__(self, seed, lr, num_epoch, device, batch_size):
        super().__init__(seed, lr, num_epoch, device, batch_size)
        self.linear = nn.Linear(27, 1)
        self.to(self.device)

class TicTacToeLR(TicTacToeMLP):
    def __init__(self, seed, lr, num_epoch, device, batch_size, hidden_layer_size=1):
        super(TicTacToeLR, self).__init__(seed, lr, num_epoch, hidden_layer_size, device, batch_size)
        self.name = "TicTacToeLR"
        self.fc1 = nn.Linear(27, 1)
        self.fc2 = None
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        self.to(device)
        self.device = device


    def forward(self, x):
        out = self.fc1(x)
        return self.sigmoid(out)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)
    



class MNISTLR(Net):
    def __init__(self, seed, lr, num_epoch, device, batch_size):
        super(MNISTLR, self).__init__(seed)
        self.name = "MNISTLR"
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.device = device

        self.fc = nn.Linear(784, 10)   # 10 classes
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        self.to(device)

    def forward(self, x):
        if type(x) == list:
            x = x[0]
        return self.fc(x)  # logits

    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch

        torch.manual_seed(self.seed)

        if not incremental:
            self.load_state_dict(copy.deepcopy(self.initial_state_dict))

        self.train(True)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fun = nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device).long()
                opt.zero_grad()
                logits = self(xb)
                loss = loss_fun(logits, yb)
                loss.backward()
                opt.step()

        self.train(False)

    def predict(self, X_test):
        torch.manual_seed(self.seed)
        self.eval()

        pred = []
        ds = torch.utils.data.TensorDataset(X_test)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device)
                logits = self(xb).to("cpu")
                y_hat = torch.argmax(logits, dim=1)
                pred.extend(y_hat.tolist())

        return np.array(pred)

    def score(self, X_test, y_test, value_functions=["accuracy"]):
        return [self._score_accuracy_gpu(X_test, y_test, self.batch_size)]

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(
            X_train_parts, y_train_parts,
            num_global_rounds=self.num_epoch,
            num_local_rounds=1,
            loss_fun=None,   
            lr=self.lr,
            test_interval=1,
            batch_size=self.batch_size,
            score=True,
            X_test=X_test,
            y_test=y_test,
            value_functions=value_functions
        )
