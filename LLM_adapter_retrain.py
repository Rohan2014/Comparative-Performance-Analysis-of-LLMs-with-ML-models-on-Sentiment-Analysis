# Description: This script fine-tunes the PEFT LoRA adpater already fine-tuned on a dataset using the PEFT method.
# Example: LLama 2.0 was trained on a specific dataset using the PEFT LoRA adapter. Now, we can fine-tune the PEFT LoRA adapter on a new dataset using this script.
import os
import numpy as np
import torch
from transformers import  DataCollatorWithPadding, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from peft import PeftModel
import evaluate
from datasets import load_dataset
import wandb
import random

# Set seed for reproducibility
seed = 42
random.seed(seed)

# Offline mode for wandb if no internet connection
os.environ["WANDB_MODE"] = "offline"

# Login to wandb
wandb.login(key="Include your key here") # API key for wandb
run = wandb.init(project='Project Name') # Check wandb documentation for more details on parameters

# HuggingFace model name
model_name = "Model Name Here"

# PEFT adapter path
adapter_path = "Fine-tuned adapter path here"+"/"

# Load the base/main model
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto")
# Use AutoModelForCausalLM for text generation tasks and remove num_labels parameter
base_model.config.pad_token_id = base_model.config.eos_token_id # ERROR FIX: Set pad_token_id to eos_token_id (Only for certain models)

# Load the base model and adapter
model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True) # Set is_trainable to True to fine-tune the adapter or False for inference

# Print the number of trainable parameters of the model
def print_number_of_trainable_model_parameters(model):
    trainable_model_parameters = 0
    all_model_parameters = 0
    for _, param in model.named_parameters():
        all_model_parameters += param.numel()
        if param.requires_grad:
            trainable_model_parameters += param.numel()
    print(f"trainable model parameters: {trainable_model_parameters}\n all model parameters: {all_model_parameters}")
    return trainable_model_parameters

# print the number of trainable parameters of PEFT adapter
original_model_parameters = print_number_of_trainable_model_parameters(model)

# Print the base/main model architecture
print("Base model Architecture:")
print(base_model)

# Print the PEFT model architecture
print("PEFT model Architecture:")
print(model)

# Load the dataset
train_data = load_dataset("csv", data_files="File Path here", split="train")
eval_data = load_dataset("csv", data_files="File Path here", split="train")
test_data = load_dataset("csv", data_files="File Path here", split="train")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id # ERROR FIX
tokenizer.pad_token = tokenizer.eos_token # ERROR FIX

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=400)

# Tokenize the dataset
tokenized_train_dataset = train_data.map(tokenize_function, batched=False)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"]) # For streamlining the dataset
tokenized_train_dataset.set_format("torch") # Not necessary

tokenized_eval_dataset = eval_data.map(tokenize_function, batched=False)
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"]) # For streamlining the dataset
tokenized_eval_dataset.set_format("torch") # Not necessary

tokenized_test_dataset = test_data.map(tokenize_function, batched=False)

# Print the tokenized datasets for verification
print("Train dataset:")
print(tokenized_train_dataset)
print(tokenized_train_dataset[0])

print("Eval dataset:")
print(tokenized_eval_dataset)
print(tokenized_eval_dataset[0])

print("Test dataset:")
print(tokenized_test_dataset)
print(tokenized_test_dataset[0])

'''

# Initialize counts for negative and positive labels
negative_count = 0
positive_count = 0

# Iterate through the 'label' column of the dataset
for label in tokenized_train_dataset['label']:
    if label == 0:
        negative_count += 1
    elif label == 1:
        positive_count += 1

pos_weights = len(tokenized_train_dataset) / (2 * positive_count)
neg_weights = len(tokenized_train_dataset) / (2 * negative_count)

'''
# Commented out the above code as it is only needed for imbalanced datasets

# Data collator for padding the tokenized datasets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metrics
def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load('precision')
    # Can load any other metric from evaluate library

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1_metric.compute(predictions=predictions, references=labels)["f1"] 
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    return {'accuracy': accuracy, 'f1_score': f1_score, 'recall': recall, 'precision': precision}

'''

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
'''
    
lr = 3e-4
batch_size = 16
num_epochs = 10

# Training arguments
training_args = TrainingArguments(
    output_dir="logs-zephyr",
    learning_rate=lr,
    lr_scheduler_type= "cosine",
    warmup_ratio= 0.05,
    max_grad_norm= 0.3,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    disable_tqdm=True,
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to="wandb"
)

# Initialize the trainer
trainer = Trainer( # Change to WeightedCELossTrainer if using custom loss function as shown above(imbalanced datasets)
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("Training model...")
trainer.train()
print("Training complete!")

# Evaluate the model on test dataset rather than running it again in inference mode
print("Predicting...")
predictions, labels_ids, metrics = trainer.predict(tokenized_test_dataset, metric_key_prefix="test")
print(metrics)
print("Predictions complete!")