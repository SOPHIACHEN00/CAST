from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from .net import AdultMLP, BankMLP, TicTacToeMLP, Dota2MLP, CreditCardMLP, MNISTMLP, IMDBMLP, TicTacToeLR, AdultLR, Dota2LR, MNISTLR
import torch

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 


def return_model(mode, seed, **kwargs):
    if mode == 'SVC':
        model = SVC(gamma=0.001)
    elif mode == 'CNNMNIST':
        model = CNNMNIST(seed=seed)

    elif mode == 'AdultMLP':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = AdultMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                         batch_size=batch_size)
    elif mode == 'BankMLP':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = BankMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                        batch_size=batch_size)
    elif mode == "Dota2MLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = Dota2MLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                         batch_size=batch_size)
    elif mode == "TicTacToeMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = TicTacToeMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                             batch_size=batch_size)
    elif mode == "CreditCardMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = CreditCardMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                             batch_size=batch_size)

    elif mode == "MNISTMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = MNISTMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                             batch_size=batch_size)
    elif mode == "IMDBMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        input_size = kwargs.get("input_size")
        model = IMDBMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                        batch_size=batch_size, input_size=input_size)
        
    elif mode == 'TicTacToeLR':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        batch_size = kwargs.get("batch_size")
        hidden_layer_size = kwargs.get("hidden_layer_size", 1)
        model = TicTacToeLR(seed=seed, lr=lr, num_epoch=num_epoch, device=device, batch_size=batch_size, hidden_layer_size=hidden_layer_size)


    elif mode == 'AdultLR':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        batch_size = kwargs.get("batch_size")
        hidden_layer_size = kwargs.get("hidden_layer_size", 1)
        model = AdultLR(seed=seed, lr=lr, num_epoch=num_epoch, device=device, batch_size=batch_size, hidden_layer_size=hidden_layer_size)

    elif mode == 'Dota2LR':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        batch_size = kwargs.get("batch_size")
        hidden_layer_size = kwargs.get("hidden_layer_size", 1)
        model = Dota2LR(seed=seed, lr=lr, num_epoch=num_epoch, device=device, batch_size=batch_size, hidden_layer_size=hidden_layer_size)


    elif mode == "MNISTLR":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        batch_size = kwargs.get("batch_size")
        model = MNISTLR(seed=seed, lr=lr, num_epoch=num_epoch, device=device, batch_size=batch_size)

    else:
        model = None
    return model
