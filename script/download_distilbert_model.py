import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def download_model(model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Downloading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    print(f"Saved successfully to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face encoder model for offline local loading."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"data/raw/hf_models/{DEFAULT_MODEL_NAME}",
        help="Directory where tokenizer/model files will be saved.",
    )
    args = parser.parse_args()

    download_model(args.model_name, Path(args.output_dir))


if __name__ == "__main__":
    main()
