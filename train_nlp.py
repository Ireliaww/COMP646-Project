# ==========================
# File: train_nlp.py
# ==========================
# %% [markdown]
# # NLP Text Preprocessing and Mapping Network Training
# This script demonstrates how to extract text features using CLIP and train a simple MLP for self-supervised reconstruction.

# %% code
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import matplotlib.pyplot as plt
from NLP_Model.text_mapping import StyleMapping

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_csv = 'Datasets/WikiArt/annotations.csv'
batch_size = 32
epochs = 10
learning_rate = 1e-3
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## 1. Text Cleaning Function

# %% code
def clean_text(text):
    # Convert to lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# %% [markdown]
# ## 2. Dataset Definition

# %% code
class TextDataset(Dataset):
    def __init__(self, csv_file, clip_model, device):
        df = pd.read_csv(csv_file)
        # Clean the text labels
        df['clean'] = df['style_label'].apply(clean_text)
        self.texts = df['clean'].tolist()
        self.clip_model = clip_model
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = self.texts[idx]
        # Tokenize and encode the text prompt
        tokens = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0)

# %% [markdown]
# ## 3. Load CLIP and DataLoader

# %% code
# Load CLIP model
clip_model, _ = clip.load('ViT-B/32', device=device)
clip_model.eval()
# Create dataset and dataloader
dataset = TextDataset(data_csv, clip_model, device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# %% [markdown]
# ## 4. Model and Optimizer Setup

# %% code
model = StyleMapping().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# %% [markdown]
# ## 5. Training Loop

# %% code
train_losses = []
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for feats in loader:
        feats = feats.to(device)
        outputs = model(feats)
        loss = loss_fn(outputs, feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

# %% [markdown]
# ## 6. Plot and Save Training Curve

# %% code
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Text Mapping Reconstruction Loss')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'nlp_training_curve.png'))
plt.close()

print("NLP training complete. Curve saved at:", os.path.join(output_dir, 'nlp_training_curve.png'))
