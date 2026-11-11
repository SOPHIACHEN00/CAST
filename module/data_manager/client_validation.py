"""Reserve deterministic client-local validation data after client partitioning."""

from __future__ import annotations

import math

import numpy as np
import torch


def reserve_client_validation(loader, *, fraction: float, seed: int) -> dict:
    """Split each training shard and keep validation data outside model training.

    Call this after the outer train/test split, any public-pool reservation, and
    the client partition. Each client receives an independent SeedSequence
    child stream. The outer test split is never read or changed.
    """
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be finite and strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if getattr(loader, "X_val_parts", None) is not None or getattr(loader, "y_val_parts", None) is not None:
        raise ValueError("client validation data has already been reserved")

    feature_parts = loader.X_train_parts
    label_parts = loader.y_train_parts
    if feature_parts is None or label_parts is None or len(feature_parts) != len(label_parts):
        raise ValueError("client training features and labels must be aligned")
    if not feature_parts:
        raise ValueError("at least one client training shard is required")

    root_sequence = np.random.SeedSequence(int(seed))
    child_sequences = root_sequence.spawn(len(feature_parts))
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    client_counts = []

    for client_id, (features, labels, child_sequence) in enumerate(
        zip(feature_parts, label_parts, child_sequences)
    ):
        if not torch.is_tensor(features) or not torch.is_tensor(labels):
            raise TypeError(f"client {client_id} shard must contain tensors")
        original_count = len(labels)
        if len(features) != original_count:
            raise ValueError(f"client {client_id} feature/label counts differ")
        if original_count < 2:
            raise ValueError(
                f"client {client_id} needs at least two samples to reserve "
                "nonempty training and validation sets"
            )

        validation_count = max(
            1, min(original_count - 1, int(math.floor(original_count * fraction + 0.5)))
        )
        client_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        permutation = np.random.default_rng(client_seed).permutation(original_count)
        validation_indices = np.sort(permutation[:validation_count])
        training_indices = np.sort(permutation[validation_count:])

        feature_training_indices = torch.as_tensor(training_indices, dtype=torch.long, device=features.device)
        feature_validation_indices = torch.as_tensor(validation_indices, dtype=torch.long, device=features.device)
        label_training_indices = feature_training_indices.to(labels.device)
        label_validation_indices = feature_validation_indices.to(labels.device)

        train_features.append(features[feature_training_indices])
        train_labels.append(labels[label_training_indices])
        val_features.append(features[feature_validation_indices])
        val_labels.append(labels[label_validation_indices])
        client_counts.append(
            {
                "client_id": client_id,
                "seed": client_seed,
                "original_count": original_count,
                "train_count": original_count - validation_count,
                "validation_count": validation_count,
            }
        )

    original_global_train_count = len(loader.y_train)
    original_partition_count = sum(row["original_count"] for row in client_counts)
    global_train_features = torch.cat(train_features, dim=0)
    global_train_labels = torch.cat(train_labels, dim=0)
    loader.X_train_parts = train_features
    loader.y_train_parts = train_labels
    loader.X_val_parts = val_features
    loader.y_val_parts = val_labels
    # Global train statistics used by some attacks must exclude validation data.
    loader.X_train = global_train_features
    loader.y_train = global_train_labels

    return {
        "fraction": float(fraction),
        "seed": int(seed),
        "original_global_train_count": original_global_train_count,
        "original_partition_count": original_partition_count,
        "unassigned_original_count": original_global_train_count - original_partition_count,
        "train_count": len(loader.y_train),
        "validation_count": sum(row["validation_count"] for row in client_counts),
        "clients": client_counts,
    }
