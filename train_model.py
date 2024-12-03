import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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


df = pd.read_csv('./datasets/processed_data.csv')
# transform the input_seq_padded column into lists not strings
df['input_seq_padded'] = df['input_seq_padded'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Create data loaders

#<<AI_ASSISTED_CODE_START>>

class NumberDataset(Dataset):
    def __init__(self, input_seqs, targets):
        self.input_seqs = input_seqs
        self.targets = targets

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.input_seqs.iloc[idx], dtype=torch.long), torch.tensor(self.targets.iloc[idx], dtype=torch.float32)

train_dataset = NumberDataset(train_data['input_seq_padded'], train_data['target'])
test_dataset = NumberDataset(test_data['input_seq_padded'], test_data['target'])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class NumberModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NumberModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output a single number
    
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        _, (h_n, _) = self.lstm(x)  # h_n: [1, batch_size, hidden_dim]
        h_n = h_n.squeeze(0)  # [batch_size, hidden_dim]
        out = self.fc(h_n)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]
    # Instantiate the model


# split dataset into training and validation sets
from sklearn.model_selection import train_test_split

vocab_size = max(vocab.values())
embedding_dim = 64
hidden_dim = 128

model = NumberModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Reduced for brevity

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(test_dataset)
    print(f"Test Loss: {avg_loss:.4f}")

    # Check some predictions
    for i in range(1):
        input_seq, target = test_dataset[i]
        output = model(input_seq.unsqueeze(0))
        predicted = output.item()
        input_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in input_seq.tolist() if idx != 0]
        print(f"Input Text: {' '.join(input_tokens)}")
        print(f"True Value: {target.item()}, Predicted Value: {predicted:.2f}\n")
    
#<<AI_ASSISTED_CODE_END>>


