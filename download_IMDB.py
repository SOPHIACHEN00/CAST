# download_imdb.py

"""
作用：
在本地下载 IMDB dataset，并保存成离线可用格式
之后可以上传到服务器，用 load_from_disk 加载
"""

from datasets import load_dataset

def main():
    print("Start downloading IMDB dataset...")

    # 下载数据（这里会自动从 Hugging Face 拉）
    dataset = load_dataset("imdb")

    print("Download finished. Saving to disk...")

    # 保存到本地
    dataset.save_to_disk("./imdb_dataset")

    print("Saved successfully to ./imdb_dataset")

if __name__ == "__main__":
    main()
