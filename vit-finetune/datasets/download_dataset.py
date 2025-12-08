import os
import argparse
import kagglehub
import shutil


def set_kagglehub_cache(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = cache_dir
    print(f"KaggleHub cache set to: {cache_dir}")


def find_dataset_root(download_path):
    for root, dirs, _ in os.walk(download_path):
        if "agricultural" in dirs:
            return root
    return None


def download_ucmerced(target_dir):
    if os.path.exists(os.path.join(target_dir, "agricultural")):
        print(f"UC Merced dataset already exists at: {target_dir}")
        return target_dir

    print("Downloading UC Merced dataset...")
    raw_path = kagglehub.dataset_download("abdulhasibuddin/uc-merced-land-use-dataset")
    print(f"Raw KaggleHub path: {raw_path}")

    dataset_root = find_dataset_root(raw_path)
    if dataset_root is None:
        raise RuntimeError("Could not locate dataset root inside KaggleHub cache.")

    print(f"Dataset found at: {dataset_root}")
    os.makedirs(target_dir, exist_ok=True)

    for folder in os.listdir(dataset_root):
        src = os.path.join(dataset_root, folder)
        dst = os.path.join(target_dir, folder)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"Dataset ready at: {target_dir}")
    return target_dir


def clear_folder(cache_dir):
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    for item in os.listdir(cache_dir):
        path = os.path.join(cache_dir, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    print(f"Cleared cache directory: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: ucmerced")
    parser.add_argument("--output", type=str, default="datasets/collections", help="Output directory")
    parser.add_argument("--cache", type=str, default="datasets/cache", help="Cache directory")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    cache_dir = os.path.abspath(args.cache)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    set_kagglehub_cache(cache_dir)

    print(f"Output Directory: {output_dir}")
    print(f"Cache Directory : {cache_dir}")

    if args.dataset.lower() == "ucmerced":
        target = os.path.join(output_dir, "ucmerced")
        download_ucmerced(target)
        clear_folder(cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
