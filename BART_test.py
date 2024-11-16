'''
AUTHOR: Lok Yee Joey Cheung
This file is used to test the built BART model by generating sample summaries. 
'''

from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch

# Load the saved model and tokenizer
model_path = "model"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Check for available device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load the test data
test = pd.read_csv('data/samsum-test.csv')

# Tokenize the test data
def tokenize_test(df):
    inputs = df["dialogue"].tolist()
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding=True, return_tensors="pt")
    return model_inputs

# Tokenize the test data
tokenized_test = tokenize_test(test)

# Move the tokenized inputs to the same device as the model
tokenized_test = {key: value.to(device) for key, value in tokenized_test.items()}

# Define batch size for summary generation
batch_size = 2  

# Generate summaries in smaller batches
generated_summaries = []
for i in range(0, len(tokenized_test["input_ids"]), batch_size):
    input_ids_batch = tokenized_test["input_ids"][i:i + batch_size]
    attention_mask_batch = tokenized_test["attention_mask"][i:i + batch_size]

    summaries_batch = model.generate(
        input_ids=input_ids_batch,
        attention_mask=attention_mask_batch,
        max_length=128,  # Adjust as needed
        num_beams=4,
        early_stopping=True
    )
    generated_summaries.extend(summaries_batch)

# Decode the generated summaries
decoded_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_summaries]

# Print or save the generated summaries
for i, summary in enumerate(decoded_summaries):
    print(f"Test Dialogue {i + 1}: {test['dialogue'].iloc[i]}")
    print(f"Generated Summary: {summary}\n")
    if i==10:
        break
