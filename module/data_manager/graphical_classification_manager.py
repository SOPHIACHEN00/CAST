import os

import sklearn.utils
import torch.utils.data
import torch
import torchvision
from sklearn.model_selection import train_test_split

from module.data_manager.manager import DataManager


# https://blog.csdn.net/sxf1061700625/article/details/105870851?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166763065216800180665746%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166763065216800180665746&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105870851-null-null.142^v63^pc_rank_34_queryrelevant25,201^v3^control_2,213^v1^control&utm_term=mnist&spm=1018.2226.3001.4187

class GraphicalClassificationManager(DataManager):
    pass

# class MNIST(GraphicalClassificationManager):
    # def __init__(self):
    #     super().__init__()
    #     return

    # def read(self, test_ratio, shuffle_seed, nrows=None):
    #     torch.manual_seed(shuffle_seed)
    #     project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     train_dataset = torchvision.datasets.MNIST(os.path.join(project_path, 'data/'),
    #                                                train=True, download=True,
    #                                                transform=torchvision.transforms.Compose([
    #                                                    torchvision.transforms.ToTensor(),
    #                                                    torchvision.transforms.Normalize(
    #                                                        (0.1307,), (0.3081,))
    #                                                ]))
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=len(train_dataset), shuffle=True)
    #     test_dataset = torchvision.datasets.MNIST(os.path.join(project_path, 'data/'),
    #                                               train=False, download=True,
    #                                               transform=torchvision.transforms.Compose([
    #                                                   torchvision.transforms.ToTensor(),
    #                                                   torchvision.transforms.Normalize(
    #                                                       (0.1307,), (0.3081,))
    #                                               ]))
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=len(test_dataset), shuffle=True)
    #     X_train, y_train = train_loader.__iter__().next()
    #     X_test, y_test = test_loader.__iter__().next()
    #     X, y = torch.concat([X_train, X_test]), torch.concat([y_train, y_test])
    #     sklearn.utils.shuffle(X, random_state=shuffle_seed)
    #     sklearn.utils.shuffle(y, random_state=shuffle_seed)

    #     if nrows is not None:
    #         X, y = X[:nrows], y[:nrows]
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         X, y, test_size=test_ratio, random_state=shuffle_seed)

    #     # self.X_train = self.X_train.reshape(len(self.X_train), 28 * 28)
    #     # self.X_test = self.X_test.reshape(len(self.X_test), 28 * 28)
    #     # self._data_to_cuda()

    #     return




# class MNIST(GraphicalClassificationManager):
#     def __init__(self):
#         super().__init__()

#     def read(self, test_ratio, shuffle_seed, nrows=None):
#         assert test_ratio is not None and test_ratio > 0, \
#             "test_ratio must be provided and > 0"

#         # reproducibility
#         g = torch.Generator()
#         g.manual_seed(shuffle_seed)

#         project_root = os.path.dirname(
#             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         )
#         root = os.path.join(project_root, "data", "raw")

#         # official TRAIN dataset ONLY
#         train_ds = torchvision.datasets.MNIST(
#             root=root,
#             train=True,
#             download=True
#         )

#         X = train_ds.data        # (60000, 28, 28), uint8
#         y = train_ds.targets     # (60000,), int64

#         # optional: quick sanity / speed test
#         if nrows is not None:
#             X = X[:nrows]
#             y = y[:nrows]

#         # ---- preprocessing ----
#         X = X.float() / 255.0
#         X = (X - 0.1307) / 0.3081     # MNIST global mean/std
#         X = X.view(X.size(0), -1)     # (N, 784) for LR / MLP
#         y = y.long()

#         # ---- split: fully controlled by test_ratio ----
#         n = X.size(0)
#         idx = torch.randperm(n, generator=g)

#         n_test = int(n * test_ratio)
#         test_idx = idx[:n_test]
#         train_idx = idx[n_test:]

#         self.X_train = X[train_idx]
#         self.y_train = y[train_idx]
#         self.X_test  = X[test_idx]
#         self.y_test  = y[test_idx]

#         return

class MNIST(GraphicalClassificationManager):
    def __init__(self):
        super().__init__()

    def read(self, test_ratio, shuffle_seed, nrows=None):
        assert test_ratio is not None and test_ratio > 0

        g = torch.Generator()
        g.manual_seed(shuffle_seed)

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        root = os.path.join(project_root, "data", "raw")


        train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True)
        test_ds  = torchvision.datasets.MNIST(root=root, train=False, download=True)

        # Merge official training & Testing dataset into one pool
        X = torch.cat([train_ds.data, test_ds.data], dim=0)      # (70000, 28, 28)
        y = torch.cat([train_ds.targets, test_ds.targets], dim=0)

        # optional speed test
        if nrows is not None:
            X = X[:nrows]
            y = y[:nrows]

        # preprocessing
        X = X.float() / 255.0
        X = (X - 0.1307) / 0.3081
        X = X.view(X.size(0), -1)    # (N, 784)
        y = y.long()

        #Unified FL split
        n = X.size(0)
        idx = torch.randperm(n, generator=g)

        n_test = int(n * test_ratio)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_test  = X[test_idx]
        self.y_test  = y[test_idx]



# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CIFAR10(GraphicalClassificationManager):
    def __init__(self):
        super().__init__()
        return

    def read(self, test_ratio, shuffle_seed, nrows=None):
        torch.manual_seed(shuffle_seed)
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, 'data/'),
                                                     train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.48836562, 0.48134598, 0.4451678),
                                                             (0.24833508, 0.24547848, 0.26617324))
                                                     ]))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=len(train_dataset), shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, 'data/'),
                                                    train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.47375134, 0.47303376, 0.42989072),
                                                            (0.25467148, 0.25240466, 0.26900575))
                                                    ]))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=len(test_dataset), shuffle=True)

        self.X_train, self.y_train = train_loader.__iter__().next()
        self.X_test, self.y_test = test_loader.__iter__().next()
