import os
from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}
    def __len__(self):
        return len(self.labels)





training_path = Path("Twitter Emotions Classification Dataset/data.csv")

df = pd.read_csv(training_path)  # or load however you want
# Assumes columns: 'text' and 'label'

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    stratify=df['label']
)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)
mini_train_dataset = EmotionDataset(train_encodings, train_labels[:2000])
mini_val_dataset = EmotionDataset(val_encodings, val_labels[:500])


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mini_train_dataset,
    eval_dataset=mini_val_dataset
)

trainer.train()

"""
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)
"""

preds = trainer.predict(val_dataset)
y_pred = preds.predictions.argmax(-1)
print(classification_report(val_labels, y_pred))

model.save_pretrained("./emotion_model")
tokenizer.save_pretrained("./emotion_model")