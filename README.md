# Code Usage Guide

## 1. Prepare the Dataset
Download the dataset you need.

## 2. Set Dataset Path
In `train.py`, set the `--data-path` argument to the **absolute path** of the extracted dataset folder.

## 3. Download Pretrained Weights
Each model defined in `model.py` includes a link to download pretrained weights. Download the weights that match the model you plan to use.

## 4. Set Weights Path
In `train.py`, set the `--weights` argument to the path of the downloaded pretrained weights.

## 5. Start Training
Once you’ve set `--data-path` and `--weights`, run `train.py` to start training.  
> The script will automatically generate a `class_indices.json` file during training.

## 6. Set Up for Prediction
In `predict.py`, import the **same model** used in training, and set the `model_weight_path` variable to the path of the trained model weights.  
> Trained weights are saved in the `weights/` directory by default.

## 7. Set Prediction Image Path
Set the `img_path` variable in `predict.py` to the **absolute path** of the image you want to predict.

## 8. Run Prediction
After setting both `model_weight_path` and `img_path`, run the `predict.py` script to make predictions.

## 9. Use a Custom Dataset
If you're using your own dataset:
- Organize it like the flower classification dataset structure: one folder per class.
- Update the `num_classes` variable in both the **training** and **prediction** scripts to match your dataset’s number of classes.
