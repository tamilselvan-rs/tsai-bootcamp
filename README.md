# tsai-bootcamp

## The Problem
Predict Handwritten digits using NN

## Dataset
[MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

| Training Data | Test Data |
| ------------- | ------------- |
| 60K  | 10K  |

## Dependencies

1. PyTorch - Base
2. Torchsummary - To review model summary
3. Torch Vision - For Transforms and datasets
4. pyplot (matplotlib) - Plotting
5. tqdm - Progress Bar

## Model
![tsainn drawio](https://github.com/tamilselvan-rs/tsai-bootcamp/assets/135374296/52f18592-7a0b-4ee3-84a1-187221b1c99b)

## Training Data Preparation
1. Inputs are scaled down to 28*28
2. Random crop is applied on 10% of the input data set
3. Random Roation is applied
4. Images are standardized by converting them to Tensor (every rgb values are mapped bewteen 0-1 without affecting the magnitude)
5. Images are normalized to avoid contrast issues
6. Training data is shuffled to avoid false accuracy due to ordering
7. Batch size is maintained at 512

## Test Data Preparation
1. Only Standardization and Normalization are applicable

## Training Steps
WIP - Need understanding on optimizers / Criterion / loss and loss.backward
## Testing Steps
WIP - Need understanding on criterion

## Results
Model Accuracy at 20 Epoch
<img width="1047" alt="Screenshot 2023-06-03 at 3 19 32 AM" src="https://github.com/tamilselvan-rs/tsai-bootcamp/assets/135374296/2cecc795-0139-4aef-8e01-12f2fccca785">
 


