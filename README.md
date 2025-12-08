# Vision Transformer (ViT) Fine-Tuning Summary

## 1. ViT Model Architecture

| Model Name | Layers | Hidden Size | MLP Size | Heads | Parameters |
|------------|--------|-------------|----------|--------|------------|
| **ViT-Base (ViT-B/16)** | 12 | 768 | 3072 | 12 | **86M** |


---

## 2. Fine-Tuning Algorithm[1]
### **Algorithm 1: Vision Transformer Fine-Tuning Procedure**

**Input:** Training images $$\{(X_i,y_i)\}_{i=n}^n$$

**Output:** Predicted labels for test set
```
   Input: Training images: {Xi, yi} n, to i=1
   Output: predicted labels of the test set.
      1. Set batchsize to 100, Optimizer Adam (learning rate: 0.0003), number of iterations to 30, image
      dimensions to 224 or 384.
      2. Set the number of mini-batches as: nb = n/batchsize
      3. For iteration = 1: Number of iterations
        3.1 For batch = 1 : nb
        • Pick a batch from the training set,
        • Generate another batch of augmented images using a particular augmentation method,
        • Train the model on the original and augmented images by minimizing the cross-entropy
        loss.
        • Backpropagate the loss.
        • Update the model parameters.
      4. Classify test image
   ```

---

## 3. Dataset Summary

| Dataset | Classes | Images per Class | Total Images | Image Size | Year |
|---------|---------|------------------|--------------|-------------|------|
| **UC Merced Land Use** | 20 | 100 | 2,000 | 256 × 256 | 2010 |

---

## 4. Results Comparison

**Train Set Used:** 30% of dataset  

| Dataset | Accuracy (ARCC Training) | Paper Accuracy [5] | Train Split |
|---------|---------------------------|--------------------|-------------|
| **UC Merced** | **96.65%** | 94.55% | 30% |

---
## 5. Result log
```
Using Python: /home/kgiri/.conda/envs/vit/bin/python
CUDA Devices: 0
Using cuda
Dataset path: datasets/collections/ucmerced
Total: 2100
Train Data: 630
Test Data: 1470
Using cached model from models/collections/vit_b_16/vit_b_16.pth
[Epoch 1/30] Train Loss = 2.5999 Test Acc = 29.84% Test Acc = 60.20%
[Epoch 2/30] Train Loss = 1.4855 Test Acc = 86.51% Test Acc = 82.45%
[Epoch 3/30] Train Loss = 0.8349 Test Acc = 93.97% Test Acc = 90.14%
[Epoch 4/30] Train Loss = 0.4378 Test Acc = 97.46% Test Acc = 92.52%
[Epoch 5/30] Train Loss = 0.2461 Test Acc = 99.52% Test Acc = 93.95%
[Epoch 6/30] Train Loss = 0.1364 Test Acc = 100.00% Test Acc = 94.63%
[Epoch 7/30] Train Loss = 0.0849 Test Acc = 100.00% Test Acc = 95.51%
[Epoch 8/30] Train Loss = 0.0583 Test Acc = 100.00% Test Acc = 95.31%
[Epoch 9/30] Train Loss = 0.0423 Test Acc = 100.00% Test Acc = 95.37%
[Epoch 10/30] Train Loss = 0.0332 Test Acc = 100.00% Test Acc = 95.65%
[Epoch 11/30] Train Loss = 0.0271 Test Acc = 100.00% Test Acc = 95.78%
[Epoch 12/30] Train Loss = 0.0233 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 13/30] Train Loss = 0.0205 Test Acc = 100.00% Test Acc = 95.71%
[Epoch 14/30] Train Loss = 0.0180 Test Acc = 100.00% Test Acc = 95.65%
[Epoch 15/30] Train Loss = 0.0163 Test Acc = 100.00% Test Acc = 95.78%
[Epoch 16/30] Train Loss = 0.0149 Test Acc = 100.00% Test Acc = 95.92%
[Epoch 17/30] Train Loss = 0.0137 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 18/30] Train Loss = 0.0127 Test Acc = 100.00% Test Acc = 95.71%
[Epoch 19/30] Train Loss = 0.0117 Test Acc = 100.00% Test Acc = 95.78%
[Epoch 20/30] Train Loss = 0.0109 Test Acc = 100.00% Test Acc = 95.71%
[Epoch 21/30] Train Loss = 0.0100 Test Acc = 100.00% Test Acc = 95.92%
[Epoch 22/30] Train Loss = 0.0094 Test Acc = 100.00% Test Acc = 95.78%
[Epoch 23/30] Train Loss = 0.0088 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 24/30] Train Loss = 0.0085 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 25/30] Train Loss = 0.0079 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 26/30] Train Loss = 0.0075 Test Acc = 100.00% Test Acc = 95.85%
[Epoch 27/30] Train Loss = 0.0071 Test Acc = 100.00% Test Acc = 95.71%
[Epoch 28/30] Train Loss = 0.0068 Test Acc = 100.00% Test Acc = 95.71%
[Epoch 29/30] Train Loss = 0.0065 Test Acc = 100.00% Test Acc = 95.78%
[Epoch 30/30] Train Loss = 0.0062 Test Acc = 100.00% Test Acc = 95.65%
Fine-tuned model saved in models/collections/vit_b_16/vit_b_16_finetuned.pth

```

## Notes
- Model: ViT-Base pretrained on ImageNet-1k.
- Fine-tuning used 224×224 images, Adam optimizer, 0.0003 learning rate, 30 iterations.
- Achieved accuracy exceeds reported values in the referenced paper.
- Paper[1] also explores more augmentation, like standard, cutmix, cutout, and hybrid, improving accuracy
- Paper[1] also tries to **Network Compression**, removing encoded layers, decreasing the training time, and increasing accuracy
    - It helps to understand what each layer is trying to attend to or learn using the attention layer
- Paper[1] also tries to use different image size ( 384 x 384) slightly increasing the accuracy.
- Paper[1] also compares the ViT models with other CNN-based models.
  

## References

[1] Bazi, Y., Bashmal, L., Rahhal, M. M. A., Dayil, R. A., & Ajlan, N. A. (2021).  
*Vision Transformers for Remote Sensing Image Classification.*  
Remote Sensing, 13(3), 516. https://doi.org/10.3390/rs1303

   
