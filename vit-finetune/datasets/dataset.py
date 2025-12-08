import torch
from torchvision import datasets, transforms
import argparse



def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def load_dataset(data_dir, size, split):
    """
    Docstring for load_dataset
    Load the dataset
    """
    #Apply transformation
    transform =  get_transforms(size)
    dataset =  datasets.ImageFolder(data_dir, transform=transform)

    #Train_Test split
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Dataset path: {data_dir}")
    print(f"Total: {len(dataset)}")
    print(f"Train Data: {train_size}")
    print(f"Test Data: {test_size}")
    return train_ds, test_ds






def main():
    parser = argparse.ArgumentParser(description= "Generate train_ds and test_ds")
    parser.add_argument("--path", type=str, required=True, help=" Dataset path")
    parser.add_argument("--image_size", type=int, required=True, help="Size of the image")
    parser.add_argument("--split", type=float, required=True, help="Train_test split")
    args = parser.parse_args()
    print(args.path)

    train_ds, test_ds = load_dataset(args.path, args.image_size, args.split)


if __name__ == "__main__":
    main()

