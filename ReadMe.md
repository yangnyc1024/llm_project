# Project Documentation

## Overview
This project focuses on fine-tuning and utilizing a pre-trained LLaMA model for sentiment analysis tasks. The process involves data preparation, prompt engineering, model fine-tuning, and prediction. This documentation outlines the environment setup, project structure, and the functionality of each script and notebook within the project.

## Environment Setup
To set up the project environment:

1. Ensure that Python and the necessary package management tools are installed on your system.
2. Navigate to the project's root directory.
3. Run the following command to install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

### Data Directory
- **/data**: This directory is intended for storing datasets used in the project. It includes training, validation, and test sets necessary for model fine-tuning and evaluation.

### Jupyter Notebooks
- **training_test.ipynb**: The core notebook for fine-tuning the LLaMA model. It includes steps for data loading, preprocessing, model training, and testing.
- **predict.ipynb**: This notebook is used for making predictions with the fine-tuned LLaMA model. It demonstrates how to load the fine-tuned model and apply it to new data for sentiment analysis.

### Main Folder
This folder contains essential Python scripts for various tasks, including model loading, fine-tuning, evaluation, and prediction.

- **evaluation.py**: Contains functions for evaluating the model's performance on a test set. Metrics include accuracy, precision, recall, F1-score, and a confusion matrix.
- **feature_engineer.py**: Provides utilities for data cleaning and feature engineering, preparing the dataset for training and testing.
- **model_base.py**: Responsible for loading the pre-trained LLaMA model from a specified model checkpoint or the Hugging Face model hub.
- **model_fine_tune.py**: Contains the logic for fine-tuning the LLaMA model on the sentiment analysis task, utilizing Parameter-Efficient Fine-Tuning (PEFT) methods.
- **predict.py**: Implements the prediction pipeline, loading the fine-tuned model and applying it to new data to infer sentiments.
- **prompt_engineer.py**: This script is used to define and modify the prompts used for training and testing the model in a zero-shot or few-shot setting.
- **train_with_fine_tuning.py**: Orchestrates the fine-tuning process, including data loading, training loop setup, model optimization, and saving the fine-tuned model.

## Usage Instructions

### Training and Testing
1. Prepare your dataset and place it in the `/data` directory.
2. Open and run the `training_test.ipynb` notebook to fine-tune the model. Follow the instructions within the notebook for step-by-step guidance.

### Prediction
1. Ensure that a model has been fine-tuned and saved appropriately.
2. Open the `predict.ipynb` notebook. This notebook will guide you through loading the fine-tuned model and applying it to new data for sentiment analysis.

### Customization
- Modify `prompt_engineer.py` to adjust the prompts used during training and testing according to your specific needs or experimentations.
- Use `feature_engineer.py` to add or modify data preprocessing and feature engineering steps.

## Contributing
Contributions to this project are welcome. Please refer to the contributing guidelines for more information on how to contribute, report issues, or request features.
