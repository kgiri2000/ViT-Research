#Fine tuning the model
import torch
import os
import torch.nn as nn
import torch.optim  as optim
from torch.utils.data import DataLoader
import argparse

"""
/project/vitftune/kgiri/VIT-RESEARCH/vit-finetune/
datasets/
    /collections/
    /cache
models/
    /collections
train.py
requirements.txt
"""
"""
Current: /project/vitftune/kgiri/VIT-RESEARCH/vit-finetune
python3 train.py 
#load_dataset
#Directory:/project/vitftune/kgiri/VIT-RESEARCH/vit-finetune/datasets 
--path collections/ucmereced
--image_size 224
--num_classes 21
--split 0.3
#Load model
Current Directory: /project/vitftune/kgiri/VIT-RESEARCH/vit-finetune/models
--model_name vit_b_16
--model_path collections/model_name
    -vit_b_16.pth
    -vit_b_16_finetuned.pth
--num_classes 21

--
"""

from datasets.dataset import load_dataset
from models.vit import load_vit

#Setup
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def fine_tune(data_dir, image_size, split,num_classes, model_name, model_path, batch_size, lr, epochs):
    train_ds, test_ds = load_dataset(data_dir, image_size, split)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    model = load_vit(model_name, model_path, num_classes).to(device)
    optimizer =  optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    #Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0,0,0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        train_acc = 100* correct/total
        train_loss = total_loss/ len(train_loader)

        #Evaluate on test set
        model.eval()
        test_correct, test_total = 0,0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                test_correct += pred.eq(y).sum().item()
                test_total += y.size(0)
        test_acc= 100* test_correct/test_total
        

        print(f"[Epoch {epoch+1}/{epochs}]"
                f" Train Loss = {train_loss:.4f}"
                f" Test Acc = {train_acc:.2f}%"
                f" Test Acc = {test_acc:.2f}%"
                )
    #Save model
    save_dir = os.path.join(model_path, model_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Fine tuned model saved in {save_path}")
    


def main():
    parser = argparse.ArgumentParser(description="For finetuning")

    # Dataset Args
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--num_classes", type=int, required=True)

    # Model Args
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    # Training Args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    fine_tune(
        data_dir=args.path,
        image_size=args.image_size,
        split=args.split,
        num_classes=args.num_classes,
        model_name=args.model_name,
        model_path=args.model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    main()

