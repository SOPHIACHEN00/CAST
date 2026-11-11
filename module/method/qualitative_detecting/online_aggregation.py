"""Paper-aligned online aggregation for WEF-family defenses."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def aggregate_online_round(
    method: str,
    base_parameters: Sequence[Sequence[torch.Tensor]],
    submitted_updates: Sequence[Sequence[torch.Tensor]],
    suspicious_clients: Sequence[int],
    device: torch.device,
) -> Dict[str, object]:
    """Aggregate submitted local models after a WEF-family decision.

    ``base_parameters[i] + submitted_updates[i]`` reconstructs client ``i``'s
    uploaded local model.  WEF keeps separate benign/free-rider model streams;
    S2-WEF uses only benign clients for the next broadcast model.
    """
    if method not in {"wef", "s2_wef"}:
        raise ValueError(f"Unsupported online aggregation method: {method}")
    if len(base_parameters) != len(submitted_updates):
        raise ValueError("Base parameters and submitted updates must be aligned")
    if not base_parameters:
        raise ValueError("At least one client is required for aggregation")

    num_clients = len(base_parameters)
    all_clients = list(range(num_clients))
    suspicious = sorted(
        client_id
        for client_id in set(suspicious_clients)
        if 0 <= int(client_id) < num_clients
    )
    suspicious_set = set(suspicious)
    benign = [client_id for client_id in all_clients if client_id not in suspicious_set]
    if not benign:
        # The papers assume a benign majority. Keep the update well-defined if
        # a detector violates that assumption in a pathological round.
        benign = all_clients
        suspicious = []
        suspicious_set = set()

    def average_local_models(client_ids: Sequence[int]) -> List[torch.Tensor]:
        averaged = [
            torch.zeros_like(parameter, device=device)
            for parameter in base_parameters[0]
        ]
        divisor = float(len(client_ids))
        for client_id in client_ids:
            client_base = base_parameters[client_id]
            client_update = submitted_updates[client_id]
            if len(client_base) != len(averaged) or len(client_update) != len(averaged):
                raise ValueError("All client parameter lists must have equal length")
            for target, base, update in zip(averaged, client_base, client_update):
                target.add_((base.to(device) + update.to(device)) / divisor)
        return averaged

    benign_parameters = average_local_models(benign)
    if method == "wef" and suspicious_set:
        suspicious_parameters = average_local_models(suspicious)
        client_parameters = [
            suspicious_parameters if client_id in suspicious_set else benign_parameters
            for client_id in all_clients
        ]
    else:
        # S2-WEF broadcasts the benign-only model to every client.
        client_parameters = [benign_parameters for _ in all_clients]

    return {
        "global_parameters": benign_parameters,
        "client_parameters": client_parameters,
        "benign_clients": benign,
        "suspicious_clients": suspicious,
    }
