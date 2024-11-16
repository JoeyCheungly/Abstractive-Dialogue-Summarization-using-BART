'''
AUTHOR: Lok Yee Joey Cheung
This file is used to fine-tune the BART-LARGE-XSUM model
'''
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline, DataCollatorForSeq2Seq
import torch
import evaluate
from datasets import Dataset
import numpy as np
import re
import nltk
nltk.download('punkt')
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from transformers import EarlyStoppingCallback
from transformers import BartConfig
import sys
import platform
import torch
import pandas as pd
import sklearn as sk


# Load the default configuration
config = BartConfig.from_pretrained('facebook/bart-large-xsum')

# Modify the configuration
config.dropout = 0.2  # Increase dropout for regularization
config.attention_dropout = 0.1  # Add attention dropout

pd.set_option('display.max_colwidth', 1000)

seed = 42

#Check device
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")

# Load dataset
train = pd.read_csv('data/samsum-train.csv')
test = pd.read_csv('data/samsum-test.csv')
val = pd.read_csv('data/samsum-validation.csv')

frames = [train, test, val]
full = pd.concat(frames)

# Remove null values
train = train.dropna()
test = test.dropna()
val = val.dropna()
full = full.dropna()

# Remove punctuations and tags
def clean_df(df, cols):
    for col in cols:
        df[col] = df[col].fillna('').apply(lambda text: re.sub('<.*?>', '', text))  # Remove tags
        df[col] = df[col].apply(lambda text: '\n'.join([line for line in text.split('\n') if not re.match('.*:\s*$', line)]))  # Remove empty dialogues
    return df

# Cleaning texts in all datasets
train = clean_df(train,['dialogue', 'summary'])
test = clean_df(test,['dialogue', 'summary'])
val = clean_df(val,['dialogue', 'summary'])

# Transforming dataframes into datasets
train_ds = Dataset.from_pandas(train)
test_ds = Dataset.from_pandas(test)
val_ds = Dataset.from_pandas(val)

# Visualizing results
print(train_ds)
print(test_ds)
print(val_ds)

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum') #Change the loaded model if necessary
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum', config=config)

# Load data collator to Prepare and batch the data (Automates padding, tensor conversion, and formatting)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

print(model)

# Reference: https://huggingface.co/docs/evaluate/transformers_integrations
def tokenization(df):
    inputs = df["dialogue"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(df["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def mapping(dataset, remove_columns):
    return dataset.map(tokenization, batched=True, remove_columns=remove_columns)

# Tokenize dataset and remove redundant columns
remove_columns = ['id', 'dialogue', 'summary']
tokenized_train = mapping(train_ds, remove_columns + ['__index_level_0__'])
tokenized_test = mapping(test_ds, remove_columns)
tokenized_val = mapping(val_ds, remove_columns)
print(tokenized_train)

# Reference: https://huggingface.co/docs/evaluate/transformers_integrations
from rouge_score import rouge_scorer

# Initialize the scorer with the types of ROUGE metrics you're interested in
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Replace -100 in labels (used for padding) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels back to txt seq
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Initialize an empty dictionary to store results
    result = {
        'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
    }

    # Compute metrics for each pair of predicted and reference sentences
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)

        # Sum up the individual precision, recall, and f1 scores
        for key in result.keys():
            result[key]['precision'] += scores[key].precision
            result[key]['recall'] += scores[key].recall
            result[key]['f1'] += scores[key].fmeasure

    # Divide by the number of examples to get the average
    num_examples = len(decoded_preds)
    for key in result.keys():
        result[key]['precision'] = round((result[key]['precision'] / num_examples) * 100, 4)
        result[key]['recall'] = round((result[key]['recall'] / num_examples) * 100, 4)
        result[key]['f1'] = round((result[key]['f1'] / num_examples) * 100, 4)
    
    print("Metrics:")
    for key, scores in result.items():
        print(f"{key} - Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F1: {scores['f1']:.4f}")


    return result


training_args = Seq2SeqTrainingArguments(
    output_dir = './results',
    eval_strategy = "epoch",
    warmup_ratio=0.1,    # Warmup for 10% of the total steps
    lr_scheduler_type="cosine",  
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=35,
    fp16=True, 
    predict_with_generate=True,
    report_to="none",
    logging_strategy="steps",  # Log at each step
    logging_steps=10,  # Log every 10 steps
    load_best_model_at_end = True,
    metric_for_best_model="eval_loss",  # Use evaluation loss as metric
    save_strategy="epoch", 
    max_grad_norm=1.0, #gradient clipping

)

# Defining Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping if no improvement for 2 evals
    )

# Train model using new compute_metric function
trainer.train()

trainer.save_model("model")

# Extract the log history from the trainer
logs = trainer.state.log_history

# Extract loss values and steps for training and validation
train_steps = []
train_losses = []
val_steps = []
val_losses = []

for log in logs:
    if 'loss' in log:  # Training loss
        train_steps.append(log['step'])
        train_losses.append(log['loss'])
    if 'eval_loss' in log:  # Validation loss
        val_steps.append(log['step'])
        val_losses.append(log['eval_loss'])

# Plot the training and validation loss
plt.plot(train_steps, train_losses, label="Training Loss")
plt.plot(val_steps, val_losses, label="Validation Loss", linestyle="--")  
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (batch_size:64)")
plt.legend()
plt.savefig('trainval_loss.png')  

