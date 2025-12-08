import os
import torch
import torch.nn as nn
import argparse
from torchvision import models


def load_vit(model_name, collns_path, num_classes):
    """
    Docstring for load_vit
    :Download and load the model
    """

    #model_path = collections/model_name
    model_path = os.path.join(collns_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    weight_file = os.path.join(model_path, f"{model_name}.pth")


    if os.path.exists(weight_file):
        print(f"Using cached model from {weight_file}")
        #Load base model
        model = models.vit_b_16(weights=None)
        #Repalace head before loading weights
        model.heads = nn.Linear(768, num_classes )
        state = torch.load(weight_file,  map_location="cpu")

        model.load_state_dict(state, strict=False)
    else:
        print("Downloading the model")
        model = models.vit_b_16(weights= models.ViT_B_16_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), weight_file)
    
        model.heads = nn.Linear(768, num_classes)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Downloading and loading model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--collns_path", type=str, required=True, help="Collection folder of models")
    parser.add_argument("--num_classes", type=int, required=True, help = "Pretuned outcome classes")
    args = parser.parse_args()
    model = load_vit(args.model_name, args.collns_path, args.num_classes)
    print("Model loaded")

if __name__ == "__main__":
    main()




