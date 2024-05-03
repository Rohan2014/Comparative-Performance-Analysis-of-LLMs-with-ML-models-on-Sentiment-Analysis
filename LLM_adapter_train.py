# Description: This script is used to fine-tune the any model on the Sentiment Analysis task using the HuggingFace Trainer API.
# Important : This script is only for models compatible with Sequence Classification heads. Check HuggingFace Model's card for compatibility.
import numpy as np
from transformers import  DataCollatorWithPadding, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig
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

# Load the base/main model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto")
# Use AutoModelForCausalLM for text generation tasks and remove num_labels parameter
model.config.pad_token_id = model.config.eos_token_id # ERROR FIX: Set pad_token_id to eos_token_id (Only for certain models)

# Print the model architecture and config
print("Base model Architecture:")
print(model)
print("Base model config:")
print(model.config)

# Function to print the number of trainable parameters of the model
def print_number_of_trainable_model_parameters(model):
    trainable_model_parameters = 0
    all_model_parameters = 0
    for _, param in model.named_parameters():
        all_model_parameters += param.numel()
        if param.requires_grad:
            trainable_model_parameters += param.numel()
    print(f"trainable model parameters: {trainable_model_parameters}\n all model parameters: {all_model_parameters}")
    return trainable_model_parameters

# print the number of trainable parameters of original model
original_model_parameters = print_number_of_trainable_model_parameters(model)

# Load the dataset
train_data = load_dataset("csv", data_files="File Path here", split="train") # Change the path to the dataset and type of dataset, if needed
eval_data = load_dataset("csv", data_files="File Path here", split="train")
test_data = load_dataset("csv", data_files="File Path here", split="train")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id # ERROR FIX
tokenizer.pad_token = tokenizer.eos_token # ERROR FIX
tokenizer.padding_side = "right" # Only for Llama 2.0 models
# Above two lines are for fixing the padding token issue for certain models

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Tokenize train, eval and test datasets
tokenized_train_dataset = train_data.map(tokenize_function, batched=False)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"]) # For streamlining the dataset

tokenized_eval_dataset = eval_data.map(tokenize_function, batched=False)
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"]) # For streamlining the dataset

tokenized_test_dataset = test_data.map(tokenize_function, batched=False)

# Print the tokenized datasets
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

# load the PEFT model
peft_config = LoraConfig(
    task_type='SEQ_CLS', 
    r=64,
    lora_alpha=16, 
    lora_dropout=0.1, 
    bias="none",
    #target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj'] # Optional or the target modules will be selected automatically
)

# Load the PEFT model
model = get_peft_model(model, peft_config)

# prints the number of trainable parameters of PEFT model
peft_model_parameters = print_number_of_trainable_model_parameters(model)
print(f"Trainable Parameters \nBefore: {original_model_parameters} \nAfter: {peft_model_parameters} \nPercentage: {round(peft_model_parameters/ original_model_parameters * 100, 2)}")

# Print the PEFT model architecture and config
print("Peft model Architecture:")
print(model)
print("Peft model config:")
print(model.config)

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
    output_dir="logs",
    learning_rate=lr,
    lr_scheduler_type= "cosine",
    warmup_ratio= 0.03,
    weight_decay=0.001,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    optim = "adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to="wandb"
)

# Initialize the trainer
trainer = Trainer( # Change the trainer to WeightedCELossTrainer if using custom loss function as shown above(imbalanced datasets)
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training model...")
trainer.train()
print("Training complete!")

# Evaluate the model but on test dataset instead of running the model again in inference mode
print("Predicting...")
predictions, labels_ids, metrics = trainer.predict(tokenized_test_dataset, metric_key_prefix="test")
print(metrics)
print("Predictions complete!")


print("Saving model...")
model.save_pretrained("Model-name") # Model name here
print("Model saved! Model-name") # Model name here for eeasy interpretation in the logs

