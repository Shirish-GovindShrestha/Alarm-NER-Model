# Alarm NER Model

A Named Entity Recognition (NER) project for an Alarm Bot, implemented in Python using Transformers. This project trains a BERT-based model to extract entities like time, date, and labels from user messages for an alarm system.

## Project Overview

- **Goal:** Extract entities from user input to set alarms accurately.
- **Model:** Huawei TinyBERT (TinyBERT_General_4L_312D) fine-tuned for NER.
- **Dataset:** Custom dataset with `tokens` and `labels` for training and testing.
- **Features:**
  - Token classification using Transformers.
  - F1, precision, and recall metrics for evaluation.
  - Training and evaluation pipelines using Hugging Face Trainer.
  - Preprocessing and tokenization with Hugging Face Tokenizers.

## Folder Structure

Alarm-NER-Model/
├── models/ # Trained model files (config.json, pytorch_model.bin)
├── dataset/ # Training and testing dataset
├── src/ # Python scripts for training and evaluation
├── README.md # Project documentation
├── requirements.txt # Python dependencies

bash
Copy code

## Installation

1. Clone the repository:

git clone https://github.com/hm/Alarm-NER-Model.git
cd Alarm-NER-Model
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Load the model

python
Copy code
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
Prepare your input and run NER using the trained model.

Optional: If the model is zipped:

python
Copy code
import zipfile

with zipfile.ZipFile("models.zip", "r") as zip_ref:
    zip_ref.extractall(".")
Training
Training is done using Hugging Face Trainer. Metrics such as precision, recall, and F1-score are computed per entity and logged per epoch. You can customize training parameters in src/train.py.

Visualization
You can use the history dictionary from training to plot:

Per-entity precision and recall per epoch.

Training loss and validation loss over epochs.

Macro F1-score trends.

python
Copy code
import matplotlib.pyplot as plt

# Example: plot training loss
plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
