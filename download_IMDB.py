# download_imdb.py

"""
Download the IMDB dataset locally and save it for offline use.
The saved directory can be transferred to a server and loaded with load_from_disk.
"""

from datasets import load_dataset

def main():
    print("Start downloading IMDB dataset...")

    # Download the dataset from Hugging Face.
    dataset = load_dataset("imdb")

    print("Download finished. Saving to disk...")

    # Save the dataset locally.
    dataset.save_to_disk("./imdb_dataset")

    print("Saved successfully to ./imdb_dataset")

if __name__ == "__main__":
    main()
