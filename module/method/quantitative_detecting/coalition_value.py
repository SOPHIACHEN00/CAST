import copy

import torch

from module.method.measure import Measure
from module.method.fl_aggregation import AggregationConfig, ServerAggregator


class AttackAwareCoalitionValue(Measure):
    """Shared coalition utility evaluation for quantitative detecting methods."""

    def __init__(
        self,
        loader,
        model,
        cache,
        value_functions,
        attacker=None,
        attacker_id=None,
        use_last_round_sv: bool = False,
        last_round_state=None,
        last_round_client_grads: dict | None = None,
        last_round_weights: list | None = None,
        base_state_dict=None,
        mini_rounds: int | None = None,
        mini_local_epochs: int | None = None,
        use_lazy: bool = False,
        lazy_honest_gap: int = 3,
        lazy_honest_phase: int = 0,
        lazy_fake_policy: str = "attack",
        lazy_attack_rounds=None,
        aggregation_config=None,
        last_round_server_state=None,
    ):
        config_dict = (
            aggregation_config.to_dict()
            if hasattr(aggregation_config, 'to_dict')
            else (aggregation_config or {})
        )
        self.aggregation_config = AggregationConfig(**config_dict)
        if self.aggregation_config.method != 'fedavg':
            if not use_last_round_sv:
                raise ValueError('New aggregation methods require last-round coalition evaluation')
            if cache:
                raise ValueError('Use a fresh coalition cache for the selected aggregation and round')
        if self.aggregation_config.method == 'fedadam' and last_round_server_state is None:
            raise ValueError('FedAdam last-round evaluation requires the pre-round server state')
        self.last_round_server_state = copy.deepcopy(last_round_server_state)
        if last_round_server_state is not None:
            ServerAggregator(self.aggregation_config, state=last_round_server_state)
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in range(len(value_functions))]

        self.attacker = attacker
        self.attacker_id = attacker_id
        self.base_state_dict = (
            copy.deepcopy(base_state_dict) if base_state_dict is not None else None
        )
        self.mini_rounds = int(
            self.model.num_epoch if mini_rounds is None else mini_rounds
        )
        self.mini_local_epochs = int(
            1 if mini_local_epochs is None else mini_local_epochs
        )

        self.use_lazy = bool(use_lazy)
        self.lazy_honest_gap = int(lazy_honest_gap)
        self.lazy_honest_phase = int(lazy_honest_phase)
        self.lazy_fake_policy = str(lazy_fake_policy)
        self.lazy_attack_rounds = (
            frozenset(int(round_num) for round_num in lazy_attack_rounds)
            if lazy_attack_rounds is not None
            else None
        )

        self.use_last_round_sv = bool(use_last_round_sv)
        self.last_round_state = last_round_state
        self.last_round_client_grads = last_round_client_grads
        self.last_round_weights = last_round_weights
        if self.use_last_round_sv:
            print(
                f"[{self.name}] use_last_round_sv=True enabled: "
                "using cached last-round state/gradients."
            )

    def evaluate_subset_world(self, parts: set, value_function: str):
        assert isinstance(parts, set)
        assert isinstance(value_function, str)
        if (
            self.attacker is not None
            and self.attacker_id is not None
            and self.attacker_id in parts
        ):
            return self.evaluate_subset_with_attack(parts, value_function)
        return self.evaluate_subset(parts, value_function)

    def _load_base_or_initial(self):
        if self.base_state_dict is not None:
            self.model.load_state_dict(copy.deepcopy(self.base_state_dict))
        elif hasattr(self.model, "initial_state_dict"):
            self.model.load_state_dict(copy.deepcopy(self.model.initial_state_dict))

    def _make_fresh_attacker(self):
        if self.attacker is None:
            return None

        attacker = copy.deepcopy(self.attacker)

        if hasattr(attacker, "global_gradient_history"):
            attacker.global_gradient_history = []
        if hasattr(attacker, "arima_started"):
            attacker.arima_started = False
        if hasattr(attacker, "arima_start_round"):
            attacker.arima_start_round = None
        if hasattr(attacker, "offline_model_trained"):
            attacker.offline_model_trained = False
        if hasattr(attacker, "offline_model"):
            attacker.offline_model = {}
        if hasattr(attacker, "_err_ema"):
            attacker._err_ema = None
        if hasattr(attacker, "_since_start"):
            attacker._since_start = 0
        if hasattr(attacker, "_last_pred_flat"):
            attacker._last_pred_flat = None
        if hasattr(attacker, "_last_pred_round"):
            attacker._last_pred_round = None

        return attacker

    def _is_lazy_honest_round(self, round_num: int) -> bool:
        if not self.use_lazy:
            return False
        if self.lazy_attack_rounds is not None:
            return int(round_num) not in self.lazy_attack_rounds
        gap = max(1, self.lazy_honest_gap)
        return ((round_num - self.lazy_honest_phase) % gap) == 0

    def _simulate_full_fl(self, parts: set, attacker_override=None):
        device = getattr(self.model, "device", torch.device("cpu"))
        plist = sorted(list(parts))
        if len(plist) == 0:
            return

        shard_sizes = torch.tensor(
            [len(self.X_train_parts[i]) for i in plist],
            dtype=torch.float,
            device=device,
        )
        weights = shard_sizes / shard_sizes.sum()

        client_models = [copy.deepcopy(self.model) for _ in plist]
        for client_model in client_models:
            client_model.to(device)

        attacker_in = (
            attacker_override is not None and self.attacker_id in plist
        )
        round_count = int(self.mini_rounds)
        local_epochs = int(self.mini_local_epochs)

        for round_num in range(round_count):
            current_round_updates = []

            for local_index, client_id in enumerate(plist):
                X_i = self.X_train_parts[client_id]
                y_i = self.y_train_parts[client_id]
                model_i = client_models[local_index].to(device)
                backup = [parameter.detach().clone() for parameter in model_i.parameters()]
                model_i.fit(
                    X_i,
                    y_i,
                    incremental=True,
                    num_epochs=local_epochs,
                )
                true_update = [
                    parameter.detach() - previous
                    for parameter, previous in zip(model_i.parameters(), backup)
                ]

                if attacker_in and client_id == self.attacker_id:
                    is_honest_round = self._is_lazy_honest_round(round_num)
                    if is_honest_round:
                        use_update = true_update
                    elif self.use_lazy and self.lazy_fake_policy == "zero":
                        use_update = [
                            torch.zeros_like(parameter, device=device)
                            for parameter in self.model.parameters()
                        ]
                    else:
                        use_update = attacker_override.get_fake_gradient(
                            round_num=round_num,
                            device=device,
                            model=model_i,
                        )
                else:
                    use_update = true_update

                current_round_updates.append(use_update)

            aggregated = [
                torch.zeros_like(parameter, device=device)
                for parameter in self.model.parameters()
            ]
            for update, weight in zip(current_round_updates, weights):
                for accumulator, parameter_update in zip(aggregated, update):
                    accumulator.add_(parameter_update, alpha=float(weight))

            if attacker_in and hasattr(attacker_override, "record_global_gradient"):
                attacker_override.record_global_gradient(
                    [gradient.detach().clone() for gradient in aggregated]
                )

            with torch.no_grad():
                for parameter, update in zip(self.model.parameters(), aggregated):
                    parameter.add_(update)

            state_dict = self.model.state_dict()
            for client_model in client_models:
                client_model.load_state_dict(state_dict)

    def _evaluate_last_round_subset(
        self,
        parts: set,
        value_function: str,
        cache_key: str,
    ):
        value_index = self.value_functions.index(value_function)
        if cache_key in self.cache:
            return self.cache[cache_key][value_index]

        if not parts:
            return self.cache[str(set())][value_index]

        if (
            self.last_round_state is None
            or self.last_round_client_grads is None
            or self.last_round_weights is None
        ):
            raise ValueError(
                "Last-round SV requested, but cached last-round "
                "state/gradients/weights are incomplete."
            )

        self.model.load_state_dict(copy.deepcopy(self.last_round_state))
        device = getattr(self.model, "device", torch.device("cpu"))
        plist = sorted(list(parts))

        subset_weights = []
        for client_id in plist:
            try:
                subset_weights.append(float(self.last_round_weights[client_id]))
            except (TypeError, IndexError, KeyError) as exc:
                raise KeyError(
                    f"Missing last-round weight for client {client_id}"
                ) from exc

        denominator = sum(subset_weights)
        if denominator <= 0:
            raise ValueError(
                "Subset weights must sum to a positive value, got "
                f"{denominator} for subset {plist}"
            )
        subset_weights = [weight / denominator for weight in subset_weights]

        parameters = list(self.model.parameters())
        updates = []
        for client_id, weight in zip(plist, subset_weights):
            if client_id not in self.last_round_client_grads:
                raise KeyError(
                    f"Missing last-round gradient for client {client_id}"
                )
            gradient_list = self.last_round_client_grads[client_id]
            updates.append([gradient.to(device) for gradient in gradient_list])

        if self.aggregation_config.method == 'fedavg':
            # Retain the existing last-round FedAvg numerical path.
            aggregated = [torch.zeros_like(parameter, device=device) for parameter in parameters]
            for gradient_list, weight in zip(updates, subset_weights):
                for accumulator, gradient in zip(aggregated, gradient_list):
                    accumulator.add_(gradient, alpha=float(weight))
        else:
            # Fresh state for EVERY coalition, including singletons and the
            # grand coalition; preceding coalition evaluations have no effect.
            server = ServerAggregator(self.aggregation_config, state=self.last_round_server_state)
            aggregated = server.step(updates, subset_weights, parameters)

        with torch.no_grad():
            for parameter, update in zip(self.model.parameters(), aggregated):
                parameter.add_(update)

        values = self.model.score(
            self.X_test,
            self.y_test,
            self.value_functions,
        )
        if cache_key not in self.cache:
            self.cache[cache_key] = [0 for _ in range(len(self.value_functions))]
        for value_index, value in enumerate(values):
            self.cache[cache_key][value_index] = value
        return self.cache[cache_key][self.value_functions.index(value_function)]

    def evaluate_subset(self, parts: set, value_function: str):
        value_index = self.value_functions.index(value_function)
        key = str(set(sorted(parts)))
        if key in self.cache:
            return self.cache[key][value_index]

        if self.use_last_round_sv:
            return self._evaluate_last_round_subset(parts, value_function, key)

        self._load_base_or_initial()
        self._simulate_full_fl(parts, attacker_override=None)

        values = self.model.score(
            self.X_test,
            self.y_test,
            self.value_functions,
        )
        if key not in self.cache:
            self.cache[key] = [0 for _ in range(len(self.value_functions))]
        for index, value in enumerate(values):
            self.cache[key][index] = value
        return self.cache[key][value_index]

    def evaluate_subset_with_attack(self, parts: set, value_function: str):
        value_index = self.value_functions.index(value_function)
        key = str(set(sorted(parts))) + "#atk"
        if key in self.cache:
            return self.cache[key][value_index]

        if self.use_last_round_sv:
            return self._evaluate_last_round_subset(parts, value_function, key)

        attacker_clone = self._make_fresh_attacker()
        self._load_base_or_initial()
        self._simulate_full_fl(parts, attacker_override=attacker_clone)

        values = self.model.score(
            self.X_test,
            self.y_test,
            self.value_functions,
        )
        if key not in self.cache:
            self.cache[key] = [0 for _ in range(len(self.value_functions))]
        for index, value in enumerate(values):
            self.cache[key][index] = value
        return self.cache[key][value_index]
