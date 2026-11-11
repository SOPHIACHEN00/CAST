import numpy as np
import sklearn.utils
import torch
from sklearn import preprocessing

from module.data_manager.manager import DataManager
from sklearn.datasets import load_diabetes, fetch_california_housing
import scipy


class StructuredRegressionManager(DataManager):
    def __init__(self):
        super().__init__()
        self.name = "StructuredRegressionManager"
        self.task = "Regression"
        return

    # use dirichlet distribution to create non-iid distributions

    def non_iid_split(self, num_parts, alpha, random_state):

        average = np.average(np.array(self.y_train))

        X_train_feature_sorted = dict(list())
        y_train_feature_sorted = dict(list())

        for i in range(len(self.y_train)):
            if self.y_train[i] >= average:
                feature = ">=average"
            else:
                feature = "<average"
            if feature not in y_train_feature_sorted.keys():
                X_train_feature_sorted[feature] = []
                y_train_feature_sorted[feature] = []
            X_train_feature_sorted[feature].append(self.X_train[i])
            y_train_feature_sorted[feature].append(self.y_train[i])

        for key in X_train_feature_sorted.keys():
            X_train_feature_sorted[key] = torch.stack(X_train_feature_sorted[key])
            y_train_feature_sorted[key] = torch.stack(y_train_feature_sorted[key])

        list_of_ratios = scipy.stats.dirichlet.rvs(alpha, size=len(X_train_feature_sorted), random_state=random_state)
        ratios = dict()
        index = 0
        for key in X_train_feature_sorted.keys():
            ratios[key] = list_of_ratios[index]
            index += 1

        self.X_train_parts = []
        self.y_train_parts = []

        lo_ratios = dict()
        for key in X_train_feature_sorted.keys():
            lo_ratios[key] = 0

        for i in range(num_parts):
            X_train_this_client = []
            y_train_this_client = []
            for label in X_train_feature_sorted.keys():
                n = len(X_train_feature_sorted[label])
                X_train_this_client.extend(X_train_feature_sorted[label][
                                           int(lo_ratios[label] * n):int((lo_ratios[label] + ratios[label][i]) * n)])
                y_train_this_client.extend(y_train_feature_sorted[label][
                                           int(lo_ratios[label] * n):int((lo_ratios[label] + ratios[label][i]) * n)])
                lo_ratios[label] += ratios[label][i]

            X_train_this_client = torch.stack(X_train_this_client)
            y_train_this_client = torch.stack(y_train_this_client)
            self.X_train_parts.append(X_train_this_client)
            self.y_train_parts.append(y_train_this_client)

        return self.X_train_parts, self.y_train_parts



    #  implementation methods：y <- max_this_client + min_this_client - y
    def flip_y_train(self, parts: set, random_seed, ratio=1):
        assert type(parts) is set
        np.random.seed(random_seed)

        for client in parts:
            # y = max + min - y
            ymax = np.max(np.array(self.y_train_parts[client]))
            ymin = np.min(np.array(self.y_train_parts[client]))
            flip = np.array([False for _ in range(len(self.y_train_parts[client]))])
            flip[:round(ratio * len(self.y_train_parts[client]))] = True
            np.random.shuffle(flip)
            self.y_train_parts[client][flip] = torch.tensor(ymax + ymin) - self.y_train_parts[client][flip]
        return


class Diabetes(StructuredRegressionManager):
    def __init__(self):
        super().__init__()
        self.name = "Diabetes"
        return

    def read(self, test_ratio, shuffle_seed, nrows=None):
        sklearn_diabetes_loader = load_diabetes(as_frame=False)
        X = sklearn_diabetes_loader.data
        y = sklearn_diabetes_loader.target

        X = np.array(X)
        y = np.array(y)
        # normalize?
        if nrows is not None:
            X = X[:nrows]
            y = y[:nrows]

        X = sklearn.utils.shuffle(X, random_state=shuffle_seed)
        y = sklearn.utils.shuffle(y, random_state=shuffle_seed)

        self.X = torch.FloatTensor(preprocessing.scale(X))
        self.y = torch.LongTensor(y)

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)
        return


    def randomly_generate_data(self, client, num_rows: int, seed):
        num_fields = len(self.X_train_parts[client][0])
        average = np.average(np.array(self.y_train_parts[client]))
        std = np.std(np.array(self.y_train_parts[client]))
        ymax = np.max(np.array(self.y_train_parts[client]))
        ymin = np.min(np.array(self.y_train_parts[client]))

        X = np.zeros(shape=(num_rows, num_fields))
        y = np.zeros(shape=num_rows)

        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        numerical_fields = np.array([0]+[i for i in range(2, 10)])
        bit_fields = np.array([1])

        bit_selection = list(set(np.array(self.X_train_parts[client]).T[1]))

        # randomly generate x - ch all numerical field
        for row in X:
            for i in numerical_fields:
                row[i] = rng.standard_normal(1)

            for i in bit_fields:
                sample = rng.standard_normal(1)
                if sample < 0:
                    row[i] = bit_selection[0]
                else:
                    row[i] = bit_selection[1]

        # randomly generate y
        for i in range(num_rows):
            sample = sum(rng.standard_normal(1))
            if type(self.y_test) is torch.LongTensor:
                y[i] = round(sample * std + average)
            else:
                y[i] = sample * std + average
            if y[i] < ymin:
                y[i] = ymin
            elif y[i] > ymax:
                y[i] = ymax

        X = torch.FloatTensor(X)
        y = torch.tensor(y)

        self.X_train_parts[client] = torch.cat([self.X_train_parts[client], X])
        self.y_train_parts[client] = torch.cat([self.y_train_parts[client], y])

        return X, y


# whole numerical field
class CaliforniaHousing(StructuredRegressionManager):
    def __init__(self):
        super().__init__()
        self.name = "CaliforniaHousing"
        return

    def read(self, test_ratio, shuffle_seed, nrows=None):
        sklearn_california_housing_loader = fetch_california_housing(as_frame=False)

        X = sklearn_california_housing_loader.data
        y = sklearn_california_housing_loader.target

        X = np.array(X)
        y = np.array(y)
        # normalize?
        if nrows is not None:
            X = X[:nrows]
            y = y[:nrows]

        X = sklearn.utils.shuffle(X, random_state=shuffle_seed)
        y = sklearn.utils.shuffle(y, random_state=shuffle_seed)

        self.X = torch.FloatTensor(preprocessing.scale(X))
        self.y = torch.tensor(y)

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        return


    def randomly_generate_data(self, client, num_rows: int, seed):
        num_fields = len(self.X_train_parts[client][0])
        average = np.average(np.array(self.y_train_parts[client]))
        std = np.std(np.array(self.y_train_parts[client]))
        ymax = np.max(np.array(self.y_train_parts[client]))
        ymin = np.min(np.array(self.y_train_parts[client]))

        X = np.zeros(shape=(num_rows, num_fields))
        y = np.zeros(shape=num_rows)

        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # randomly generate x - ch all numerical field
        for row in X:
            for i in range(num_fields):
                row[i] = rng.standard_normal(1)

        # randomly generate y
        for i in range(num_rows):
            sample = sum(rng.standard_normal(1))
            if type(self.y_test) is torch.LongTensor:
                y[i] = round(sample * std + average)
            else:
                y[i] = sample * std + average
            if y[i] < ymin:
                y[i] = ymin
            elif y[i] > ymax:
                y[i] = ymax

        X = torch.FloatTensor(X)
        y = torch.tensor(y)

        self.X_train_parts[client] = torch.cat([self.X_train_parts[client], X])
        self.y_train_parts[client] = torch.cat([self.y_train_parts[client], y])

        return X, y
