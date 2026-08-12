import os

if __name__ == "__main__":
    import nltk
    user = os.getenv("USER") # downloading on capstor/iopstor will give permission error
    data_dir = os.getenv("NLTK_DATA_DIR", f"/users/{user}/nltk_data")
    data_dir = os.path.realpath(data_dir)
    os.makedirs(data_dir, exist_ok=True) 
    print(data_dir)
    pkgs = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")
    [nltk.download(pkg, download_dir=data_dir, quiet=True) for pkg in pkgs]