import argparse
import os
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


DEFAULT_CHECKPOINT = "distilbert-base-uncased"


def encode_texts(texts, tokenizer, encoder, device, batch_size, max_length):
    features = []
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_texts in loader:
            encoded = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if "token_type_ids" in encoded:
                encoded.pop("token_type_ids")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = encoder(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
            features.append(cls_embeddings)

    return torch.cat(features, dim=0).float()


def main():
    parser = argparse.ArgumentParser(
        description="Precompute IMDB DistilBERT embeddings once for repeated offline FL experiments."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/raw/imdb_dataset",
        help="Path to the downloaded IMDB Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=f"data/raw/hf_models/{DEFAULT_CHECKPOINT}",
        help="Path to the downloaded local Hugging Face encoder directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint name, recorded in metadata only.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"data/raw/imdb_preprocessed/imdb_{DEFAULT_CHECKPOINT}_maxlen256.pt",
        help="Output .pt file containing precomputed embeddings and labels.",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer truncation length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Encoding batch size.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_path)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"IMDB dataset directory not found: {dataset_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")

    print(f"Loading IMDB dataset from: {dataset_dir.resolve()}")
    raw = load_from_disk(str(dataset_dir))
    if "unsupervised" in raw:
        del raw["unsupervised"]

    pool = concatenate_datasets([raw["train"], raw["test"]])
    texts = pool["text"]
    labels = torch.tensor(pool["label"], dtype=torch.long)

    print(f"Loading local encoder from: {model_dir.resolve()}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
    encoder.eval()

    print(f"Encoding {len(texts)} IMDB samples on device: {device}")
    X = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    payload = {
        "X": X,
        "y": labels,
        "checkpoint": args.checkpoint,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_samples": len(labels),
        "embedding_dim": int(X.shape[1]),
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))

    print(f"Saved preprocessed IMDB embeddings to: {output_path.resolve()}")
    print(f"num_samples   : {payload['num_samples']}")
    print(f"embedding_dim : {payload['embedding_dim']}")


if __name__ == "__main__":
    main()
