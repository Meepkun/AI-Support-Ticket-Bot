import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from models.bert_ticket_model import TicketClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('data/tickets.csv')

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# -----------------------------
# Tokenize tickets
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(list(df['ticket']),
                   max_length=64,
                   padding=True,
                   truncation=True,
                   return_tensors='pt')

# -----------------------------
# Prepare dataset & dataloader
# -----------------------------
labels = torch.tensor(df['label_enc'].values)
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# Initialize model
# -----------------------------
n_classes = len(le.classes_)
model = TicketClassifier(n_classes)

# Optimizer and loss
optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

# -----------------------------
# Training loop
# -----------------------------
model.train()
for epoch in range(3):
    for batch in loader:
        input_ids, attention_mask, targets = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# -----------------------------
# Define category responses
# -----------------------------
responses = {
    "Password/Account Recovery": "Please click on ‘Forgot Password’ on the login page and follow the instructions to reset your password.",
    "Leave/HR Queries": "You can check your leave balance from the HR portal under ‘My Leave’ section."
}

# -----------------------------
# Function to classify & reply
# -----------------------------
def reply_ticket(ticket_text):
    model.eval()
    tokens = tokenizer(ticket_text, max_length=64, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(tokens['input_ids'], tokens['attention_mask'])
        pred = torch.argmax(outputs, dim=1).item()
    category = le.inverse_transform([pred])[0]
    return f"Category: {category}\nResponse: {responses[category]}"

# -----------------------------
# Test
# -----------------------------
test_tickets = [
    "Forgot my password",
    "Unable to log in",
    "Where can I see my leave?"
]

for t in test_tickets:
    print(reply_ticket(t))
