# train_cnn.py

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import config
from preprocess import get_cnn_datasets

# --- Định nghĩa Lớp Model và Dataset ---
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout_rate, padding_idx=0):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, hidden_size, ks) for ks in [3, 4, 5]])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(self.convs) * hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = [self.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class TextDataset(Dataset):
    def __init__(self, tokens_series, labels_series, vocab_map, max_seq_len):
        self.labels = labels_series.tolist()
        self.texts_ids = [self._text_to_ids(tokens, vocab_map, max_seq_len) for tokens in tokens_series]

    def _text_to_ids(self, text_tokens, vocab_map, max_seq_len):
        ids = [vocab_map.get(token, 0) for token in text_tokens] # 0 là UNK_ID
        ids = ids[:max_seq_len] if len(ids) > max_seq_len else ids + [0] * (max_seq_len - len(ids)) # 0 là PAD_ID
        return ids

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return torch.tensor(self.texts_ids[idx]), torch.tensor(self.labels[idx])

# --- Các hàm Huấn luyện và Đánh giá ---
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for text, label in dataloader:
        text, label = text.to(config.DEVICE), label.to(config.DEVICE)
        optimizer.zero_grad()
        pred = model(text)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, label in dataloader:
            text, label = text.to(config.DEVICE), label.to(config.DEVICE)
            pred = model(text)
            loss = criterion(pred, label)
            total_loss += loss.item()
            all_preds.extend(pred.argmax(1).cpu().tolist())
            all_labels.extend(label.cpu().tolist())
    return total_loss / len(dataloader), all_preds, all_labels

# --- Hàm chính để chạy thử nghiệm CNN ---
def train_cnn_model(df, le, num_classes):
    print("\n--- Bắt đầu Huấn luyện Mô hình CNN ---")
    run_name = "CNN_FastText_Lem"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n[MLflow Run] {run_name}")
        params = {
            "model_type": "CNN", "embedding": "FastText", "preprocessing": "Lemmatization",
            "embedding_dim": config.CNN_EMBEDDING_DIM, "hidden_size": config.CNN_HIDDEN_SIZE,
            "max_seq_length": config.CNN_MAX_SEQ_LENGTH, "dropout_rate": config.CNN_DROPOUT_RATE,
            "epochs": config.CNN_EPOCHS, "batch_size": config.CNN_BATCH_SIZE, "learning_rate": config.CNN_LEARNING_RATE,
        }
        mlflow.log_params(params)
        
        train_tokens, test_tokens, y_train, y_test, ft_model = get_cnn_datasets(df)
        
        vocab_map = ft_model.wv.key_to_index
        vocab_size = len(ft_model.wv)
        
        train_dataset = TextDataset(train_tokens, y_train, vocab_map, config.CNN_MAX_SEQ_LENGTH)
        test_dataset = TextDataset(test_tokens, y_test, vocab_map, config.CNN_MAX_SEQ_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.CNN_BATCH_SIZE)
        
        model = CNNClassifier(vocab_size, config.CNN_EMBEDDING_DIM, config.CNN_HIDDEN_SIZE,
                              num_classes, config.CNN_DROPOUT_RATE).to(config.DEVICE)
        
        # Load pre-trained weights
        model.embedding.weight.data.copy_(torch.from_numpy(ft_model.wv.vectors))
        print("Đã tải trọng số FastText vào lớp Embedding.")

        optimizer = optim.Adam(model.parameters(), lr=config.CNN_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(config.CNN_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            test_loss, _, _ = evaluate(model, test_loader, criterion)
            print(f"Epoch {epoch+1}/{config.CNN_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "test_loss": test_loss}, step=epoch)

        _, y_pred, y_true = evaluate(model, test_loader, criterion)
        
        # Log metrics và confusion matrix
        from train_ml import compute_metrics
        compute_metrics(y_true, y_pred, run_name, le, num_classes)
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        print(f"Đã hoàn tất run: {run_name}")