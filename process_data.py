import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# <<AI_ASSISTED_CODE_START>>
vocab = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100,
    'thousand': 1000,
    'and': 0  # 'and' can be mapped to 0 or ignored
}

df = pd.read_csv('./datasets/all_data.csv')

# Function to map words to their numerical values
def text_to_numbers(text):
    tokens = text.split()
    numbers = []
    for token in tokens:
        if token in vocab:
            numbers.append(vocab[token])
        else:
            # Handle any unexpected tokens
            numbers.append(0)
    return numbers

# Apply the function to create input sequences
df['input_seq'] = df['text'].apply(text_to_numbers)

# The target outputs are the actual numbers
df['target'] = df['number'].astype(float)  # Convert to float for regression

# Display the processed data
print(df[['text', 'input_seq', 'target']].head(10))

# <<AI_ASSISTED_CODE_END>>

# 1) convert input_seq to list of integers and get maximum seq length so all can be the same length
df['input_seq'] = df['input_seq'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
max_len = df['input_seq'].apply(len).max()

def pad_the_sequence(seq, max_len):
    return list(seq + list([0]*(max_len - len(seq))))

# alter the dataframe to include the padded sequences
df['input_seq_padded'] = df['input_seq'].apply(lambda x: pad_the_sequence(x, max_len))

df.to_csv('./datasets/processed_data.csv', index=False)


df.to_csv('processed_data.csv', index=False)