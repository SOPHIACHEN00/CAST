import time
from itertools import combinations as cbs
from math import comb

import torch

from module.method.quantitative_detecting.coalition_value import (
    AttackAwareCoalitionValue,
)


class ShapleyValue(AttackAwareCoalitionValue):
    name = "ShapleyValue"

    def phi(self, i, value_function):
        accu = 0.0
        all_set = set(range(self.num_parts))
        all_set.remove(i)
        for S_size in range(len(all_set) + 1):
            weight = 1.0 / self.num_parts / comb(self.num_parts - 1, S_size)
            for cb in cbs(all_set, S_size):
                set_with_i = set(cb)
                set_with_i.add(i)
                set_without_i = set(cb)
                accu += (
                    self.evaluate_subset_world(set_with_i, value_function)
                    - self.evaluate_subset_world(set_without_i, value_function)
                ) * weight
        return accu

    def get_contributions(self, **kwargs):
        device = self.model.device
        t0 = time.time()

        for idx_val in range(len(self.value_functions)):
            value_function = self.value_functions[idx_val]
            for i in range(self.num_parts):
                self.contributions[idx_val][i] = self.phi(i, value_function)

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0
        return self.contributions.tolist(), t_cal
