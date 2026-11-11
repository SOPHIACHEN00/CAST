import copy
import itertools
from collections import defaultdict, Counter
from math import floor

import numpy as np
import torch

import config
from typing import Dict
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from itertools import combinations as cbs
from abc import ABC, abstractmethod
from module.data_manager import DataManager


class Measure(ABC):

    def __init__(self, loader: DataManager, model, cache: Dict, value_functions) -> None:

        self.cache = cache
        self.name = 'None'
        self.X_train_parts = copy.deepcopy(loader.X_train_parts)
        self.y_train_parts = copy.deepcopy(loader.y_train_parts)
        self.X_test = copy.deepcopy(loader.X_test)
        self.y_test = copy.deepcopy(loader.y_test)
        self.model = model
        self.num_parts = len(self.X_train_parts)
        self.value_functions = value_functions
        self.contributions = np.zeros([len(self.value_functions), self.num_parts])
        for idx_val, i in itertools.product(range(self.contributions.shape[0]), range(self.contributions.shape[1])):
            self.contributions[idx_val][i] = -float(np.inf)

        # 
        if "gradient_similarity" in value_functions:
            self.gradient_similarity_cache = dict()

        return

    def __str__(self):
        return self.name

    @abstractmethod
    def get_contributions(self, **kwargs):
        pass

    # self.caches = list of dict
    def evaluate_subset(self, parts: set, value_function: str, **kwargs):
        assert type(value_function) is str
        assert type(parts) is set
        # print(f"evaluating {parts}")
        function_index = self.value_functions.index(value_function)
        parts = set(sorted(parts))
        # compute v(S)
        if self.cache.get(str(parts)) is not None:
            val_list = self.cache[str(parts)]

        # If not in the cache
        else:
            # value functions all belongs to model performance!
            if set([self.value_functions[i] in ["f1", "f1_macro", "f1_micro", "accuracy"] for i in
                    range(len(self.value_functions))]) == {True}:
                X_train_parts_coal = copy.deepcopy([self.X_train_parts[i] for i in parts])
                y_train_parts_coal = copy.deepcopy([self.y_train_parts[i] for i in parts])
                # refit the model and get the scores of value functions!
                val_list = self.model.fed_train_and_score(X_train_parts_coal, y_train_parts_coal, self.X_test,
                                                          self.y_test,
                                                          self.value_functions)
                # val_list = self.model.fit_and_score(X_train_coal, y_train_coal, self.X_test, self.y_test,
                #                                     self.value_functions)

                # self.model.fit(X_train_coal, y_train_coal)
                # y_pred = self.model.predict(self.X_test)

            # if none of the value functions is model performance
            else:
                val_list = []
                for value_function in self.value_functions:
                    if value_function == "robust_volume":
                        omega = kwargs.get("omega", 0.1)
                        v = self.compute_robust_volumes(parts, omega=omega, device=self.model.device)
                    elif value_function == "gradient_similarity":
                        # gs_alpha = 0.95, num_local_epochs = 1
                        gs_alpha = kwargs.get("gs_alpha", 0.95)
                        num_local_epochs = kwargs.get("num_local_epochs", config.num_local_epochs)
                        v = self.compute_gradient_similarity(parts, gs_alpha=gs_alpha,
                                                             num_local_epochs=num_local_epochs)
                    elif value_function == "data_quantity":
                        v = self.compute_data_quantity(parts, )
                    else:
                        raise Exception(f"Value function {value_function} does not exist!")
                    val_list.append(v)

            # If it is not in the cache, maintain the cache.
            self.cache[str(parts)] = copy.deepcopy(val_list)
        return val_list[function_index]

    def get_remove_client_data(self):
        # Check whether scores have been generated
        for idx_val, i in itertools.product(range(self.contributions.shape[0]), range(self.contributions.shape[1])):
            if self.contributions[idx_val][i] != -float(np.inf):
                break
        else:
            raise Exception("scores not generated!")

        y_remove_best = np.zeros([len(self.value_functions), self.num_parts])
        y_remove_worst = np.zeros([len(self.value_functions), self.num_parts])
        num_removed_client_best = np.zeros([len(self.value_functions), self.num_parts])
        num_removed_client_worst = np.zeros([len(self.value_functions), self.num_parts])
        for idx_val in range(len(self.value_functions)):

            # reverse == True means remove best
            for reverse, y, num_removed_client in zip([True, False], [y_remove_best[idx_val], y_remove_worst[idx_val]],
                                                      [num_removed_client_best[idx_val],
                                                       num_removed_client_worst[idx_val]]):
                # Construct the correspondence between the source and the fraction
                source_credit_pair = [Credit(i, self.contributions[idx_val][i]) for i in range(self.num_parts)]
                source_credit_pair.sort(reverse=reverse)
                retained_clients = set(range(self.num_parts))

                for i in range(self.num_parts):
                    y[i] = self.evaluate_subset(retained_clients, self.value_functions[idx_val])
                    # Eliminate the remaining best (bad) participants at present
                    retained_clients.remove(source_credit_pair[i].source)
                    num_removed_client[i] = source_credit_pair[i].source

        # Eliminate the remaining best (bad) participants at present
        removed_ratio = np.linspace(0, 1, self.num_parts + 1)
        removed_ratio = [round(e, 5) for e in removed_ratio]
        removed_ratio = removed_ratio[0:-1]

        return removed_ratio, y_remove_best, y_remove_worst, num_removed_client_best, num_removed_client_worst

    # --------------gradient similarity begins----------------------
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

    def cosine_similarity(self, grad1, grad2, normalized=False):
        """
        Input: two sets of gradients of the same shape
        Output range: [-1, 1]
        """
        cos_sim = F.cosine_similarity(self.flatten(grad1), self.flatten(grad2), 0, 1e-10)
        if normalized:
            return (cos_sim + 1) / 2.0
        else:
            return cos_sim

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

    # hyperparams: gs_alpha = 0.95, num_local_epochs = 1, gamma_gs = 0.5, gamma_scheduler = 0.9
    def compute_gradient_similarity(self, parts, gs_alpha, num_local_epochs):
        assert type(parts) is set
        parts = set(sorted(parts))

        if len(self.gradient_similarity_cache) != 0:
            return self.gradient_similarity_cache[str(parts)]

        self.gradient_similarity_cache[str(set())] = 0

        for S_size in range(1, self.num_parts + 1):
            for cb in cbs(list(range(self.num_parts)), S_size):
                sorted_cb = set(sorted(set(cb)))
                self.gradient_similarity_cache[str(sorted_cb)] = np.inf

        shard_sizes = [len(self.X_train_parts[i]) for i in range(self.num_parts)]
        shard_sizes = torch.tensor(shard_sizes).float()
        # Initialize the model
        model_architecture = self.model.__class__

        seed = self.model.seed
        lr = self.model.lr
        num_epoch = self.model.num_epoch
        hidden_layer_size = self.model.hidden_layer_size
        device = self.model.device
        batch_size = self.model.batch_size

        # ---- init the clients ----
        client_models = []
        # server_model 定为 self.model
        self.model.load_state_dict(self.model.initial_state_dict)

        for i in range(self.num_parts):
            model = copy.deepcopy(self.model)
            client_models.append(model)

        # ---- CML/FL begins ---- 
        for iteration in range(num_epoch):
            gradients = []
            for i in range(self.num_parts):
                X_i, y_i = self.X_train_parts[i], self.y_train_parts[i]
                model = client_models[i]
                # model.train()
                model = model.to(device)
                backup = copy.deepcopy(model)
                model.fit(X_i, y_i, incremental=True, num_epochs=num_local_epochs)
                gradient = self.compute_grad_update(old_model=backup, new_model=model, device=device)
                gradients.append(gradient)

            # until now, all the normalized (based on gamma_gs) gradients of clients have been generated and
            # stored in gradients

            # ---- Server Aggregate ----
            weights = torch.div(shard_sizes, torch.sum(shard_sizes))
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]
            for gradient, weight in zip(gradients, weights):
                self.add_gradient_updates(aggregated_gradient, gradient, weight=weight)

            self.add_update_to_model(self.model, aggregated_gradient)
            # update reputation and calculate reward gradients
            flat_aggre_grad = self.flatten(aggregated_gradient)

            # aggregated_gradient and gradients are ready!
            for S_size in range(1, self.num_parts + 1):
                for cb in cbs(list(range(self.num_parts)), S_size):
                    sorted_cb = set(sorted(set(cb)))
                    cb_shard_sizes = torch.tensor([len(self.X_train_parts[i]) for i in sorted_cb])
                    cb_weights = torch.div(cb_shard_sizes, torch.sum(cb_shard_sizes))
                    cb_gradients = [gradients[i] for i in sorted_cb]
                    cb_aggre_gradient = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]
                    for cb_gradient, cb_weight in zip(cb_gradients, cb_weights):
                        self.add_gradient_updates(cb_aggre_gradient, cb_gradient, weight=cb_weight)

                    cg = F.cosine_similarity(self.flatten(cb_aggre_gradient), flat_aggre_grad, 0, 1e-10)
                    if self.gradient_similarity_cache[str(sorted_cb)] == np.inf:
                        self.gradient_similarity_cache[str(sorted_cb)] = cg
                    else:
                        self.gradient_similarity_cache[str(sorted_cb)] = gs_alpha * self.gradient_similarity_cache[
                            str(sorted_cb)] + (1 - gs_alpha) * cg

            # ------------ server broadcast -----------
            for i in range(self.num_parts):
                client_models[i].load_state_dict(self.model.state_dict())

        # training finished!
        y_pred = self.model.predict(self.X_test)
        print(f"accuracy of CG global model: {accuracy_score(self.y_test, y_pred)}")

        return self.gradient_similarity_cache[str(parts)]

    # -------------------------- gradient similarity ends -------------------

    def compute_data_quantity(self, parts):
        assert type(parts) is set
        parts = set(sorted(parts))
        X_S = torch.cat([self.X_train_parts[i] for i in parts])
        return len(X_S)

    # -------------- robust volume begins -----------
    #
    def compute_robust_volumes(self, parts, omega, device):
        assert type(parts) is set
        parts = set(sorted(parts))
        X_S = torch.cat([self.X_train_parts[i] for i in parts])

        X_tilde, cubes = self.compute_X_tilde_and_counts(X_S, omega)
        X_tilde = np.array(X_tilde).astype(np.float64)
        X_tilde = torch.tensor(X_tilde)

        X_tildes, dcube_collections = [X_tilde], [cubes]
        N = sum([len(X_tilde) for X_tilde in X_tildes])

        alpha = 1.0 / (10 * N)  # it means we set beta = 10

        # volumes, volume_all = self.compute_volumes(X_tildes, d=X_tildes[0].shape[1])
        volumes = self.compute_volumes(X_tildes, device, d=X_tildes[0].shape[1])
        robust_volumes = np.zeros_like(volumes)

        for i, (volume, hypercubes) in enumerate(zip(volumes, dcube_collections)):
            rho_omega_prod = 1.0
            for cube_index, freq_count in hypercubes.items():
                # if freq_count == 1: continue # volume does not monotonically increase with omega
                # commenting this if will result in volume monotonically increasing with omega
                rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

                rho_omega_prod *= rho_omega

            robust_volumes[i] = (volume * rho_omega_prod)

            return robust_volumes[0]

    # Vol
    def compute_volumes(self, datasets, device, d):

        # d = datasets[0].shape[1]

        # for i in range(len(datasets)):
        #     datasets[i] = datasets[i].reshape(-1 ,d)

        # X = np.concatenate(datasets, axis=0).reshape(-1, d)

        volumes = np.zeros(len(datasets))
        for i, dataset in enumerate(datasets):

            dataset.to(device)
            dataset = np.round(dataset, 3)
            mul_res = torch.matmul(dataset.T, dataset)

            mul_res = np.array(mul_res)

            mul_res_max_elem = np.max(mul_res)
            normalized_mul_res = mul_res / mul_res_max_elem
            normalized_det_res = np.linalg.det(normalized_mul_res)

            # print(normalized_det_res)

            det_res = normalized_det_res * (mul_res_max_elem ** d)
            if det_res < 0:
                det_res = 0
            volumes[i] = np.sqrt(det_res)
        return volumes

    def compute_X_tilde_and_counts(self, X: torch.tensor, omega):
        """
        Compresses the original feature matrix X to  X_tilde with the specified omega.
        Returns:
        X_tilde: compressed np.ndarray
        cubes: a dictionary of cubes with the respective counts in each dcube
        """
        # omega = self.omega
        # D = X.shape[1]

        # assert 0 < omega <= 1, "omega must be within range [0,1]."

        # m = ceil(1.0 / omega) # number of intervals for each dimension

        cubes = Counter()  # a dictionary to store the freqs
        # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
        # value: counts

        Omega = defaultdict(list)
        # Omega = {}
        # a dictionary to store cubes of not full size

        min_ds = torch.min(X, dim=0).values

        for x in X:
            cube = np.zeros_like(x)
            for d, xd in enumerate(x - min_ds):
                d_index = floor(xd / omega) 
                cube[d] = d_index

            cube_key = tuple(cube)  
            cubes[str(cube_key)] += 1

            Omega[str(cube_key)].append(x)

            '''
            if cube_key in Omega:

                # Implementing mean() to compute the average of all rows which fall in the cube

                Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
                # Omega[cube_key].append(x)
            else:
                Omega[cube_key] = x
            '''

        X_tilde = torch.stack([torch.stack(list(value)).float().mean(dim=0) for key, value in Omega.items()])

        # X_tilde = stack(list(Omega.values()))
        # Vol(X_tilde) × Pi i∈Ψ ρi
        return X_tilde, cubes

 

class DataQuantity(Measure):
    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions)
        self.name = "DataQuantity"
        self.contributions = None 

    def compute_contributions(self):
        contributions = [self.compute_data_quantity({i}) for i in range(self.num_parts)]
        self.contributions = contributions 
        return contributions  # return list

    def get_contributions(self, **kwargs):
        if self.contributions is None:
            self.compute_contributions() 
        return self.contributions, None  # Compatible with score_clients() format

    def get_remove_client_data(self):
        if self.contributions is None:
            raise Exception("scores not generated!")

        x = list(range(len(self.contributions)))
        y_remove_best = sorted(self.contributions, reverse=True)
        y_remove_worst = sorted(self.contributions)

        num_removed_client_best = list(range(1, len(self.contributions) + 1))
        num_removed_client_worst = list(range(1, len(self.contributions) + 1))

        return x, y_remove_best, y_remove_worst, num_removed_client_best, num_removed_client_worst



class RobustVolume(Measure):
    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions)
        self.name = "RobustVolume"
        self.contributions = None  

    def compute_contributions(self, omega=0.1):
        contributions = [self.compute_robust_volumes({i}, omega=omega, device=self.model.device) for i in range(self.num_parts)]
        self.contributions = contributions  
        return contributions  

    def get_contributions(self, **kwargs):
        if self.contributions is None:
            self.compute_contributions()
        return self.contributions, None 

    def get_remove_client_data(self):
        if self.contributions is None:
            raise Exception("scores not generated!")

        x = list(range(len(self.contributions)))
        y_remove_best = sorted(self.contributions, reverse=True)
        y_remove_worst = sorted(self.contributions)

        num_removed_client_best = list(range(1, len(self.contributions) + 1))
        num_removed_client_worst = list(range(1, len(self.contributions) + 1))

        return x, y_remove_best, y_remove_worst, num_removed_client_best, num_removed_client_worst


class GradientSimilarity(Measure):
    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions)
        self.name = "GradientSimilarity"
        self.gradient_similarity_cache = {}  
        self.contributions = None 

    def compute_contributions(self, gs_alpha=0.95, num_local_epochs=1):
        contributions = [self.compute_gradient_similarity({i}, gs_alpha=gs_alpha, num_local_epochs=num_local_epochs)
                         for i in range(self.num_parts)]
        self.contributions = contributions 
        return contributions  

    def get_contributions(self, **kwargs):
        if self.contributions is None:
            self.compute_contributions()
        return self.contributions, None

    def get_remove_client_data(self):
        if self.contributions is None:
            raise Exception("scores not generated!")

        x = list(range(len(self.contributions)))
        y_remove_best = sorted(self.contributions, reverse=True)
        y_remove_worst = sorted(self.contributions)

        num_removed_client_best = list(range(1, len(self.contributions) + 1))
        num_removed_client_worst = list(range(1, len(self.contributions) + 1))

        return x, y_remove_best, y_remove_worst, num_removed_client_best, num_removed_client_worst






class Credit(object):
    def __init__(self, source, credit):
        self.source = source
        self.credit = credit
        return

    def __gt__(self, other):
        return self.credit > other.credit

    def __lt__(self, other):
        return self.credit < other.credit


# NEW! gradient attack
class GradientReplayAttack:
    def __init__(self, k, start_attack_round):
        self.k = k 
        self.start_attack_round = start_attack_round  # start-attacking
        self.global_gradient_history = []

    def get_fake_gradient(self, round_num, device, model):
        if round_num < self.start_attack_round or len(self.global_gradient_history) < self.k:
            # first M rounds：randomly generare gradient
            return [torch.randn_like(param) for param in model.parameters()]
        else: # calculate avg value
            avg_gradient = [
                sum(t[i] for t in self.global_gradient_history[-self.k:]) / self.k
                for i in range(len(self.global_gradient_history[0]))
            ]
            return avg_gradient

    def record_global_gradient(self, global_gradient):
        self.global_gradient_history.append(global_gradient)
