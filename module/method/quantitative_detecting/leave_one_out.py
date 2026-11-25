import time

import torch

from .coalition_value import AttackAwareCoalitionValue


class LeaveOneOut(AttackAwareCoalitionValue):
    """Estimate each client's contribution as v(N) minus v(N without that client)."""

    name = "LeaveOneOut"

    def get_contributions(self):
        started_at = time.perf_counter()
        clients = set(range(self.num_parts))
        for value_index, value_function in enumerate(self.value_functions):
            grand_value = self.evaluate_subset_world(clients, value_function)
            for client in range(self.num_parts):
                without_client = self.evaluate_subset_world(clients - {client}, value_function)
                self.contributions[value_index, client] = grand_value - without_client

        device = torch.device(self.model.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return self.contributions.tolist(), time.perf_counter() - started_at
