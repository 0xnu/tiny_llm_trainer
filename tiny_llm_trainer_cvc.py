import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from torchaudio import transforms as T

@dataclass
class CVCLionHeart:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    dropout: float

class CVCModel(nn.Module):
    def __init__(self, config: CVCLionHeart):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
            ),
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.embed_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 80)  ## Output is mel spectrogram frames with 80 mel bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.relu(self.linear(x))
        return self.output(x)

class CVCDataset(Dataset):
    def __init__(self, root_dir: str, tsv_file: str, vocab: Dict[str, int], max_len: int = 200):
        self.root_dir = root_dir
        self.data = self._load_tsv(os.path.join(root_dir, tsv_file))
        self.vocab = vocab
        self.max_len = max_len
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
        )

    def _load_tsv(self, tsv_path: str) -> List[Tuple[str, str]]:
        data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            next(f)  ## Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    data.append((parts[1], parts[0]))  ## (text, audio_path)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, audio_path = self.data[idx]
        
        ## Text processing
        tokens = [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        tokens = tokens[:self.max_len] + [self.vocab['<PAD>']] * (self.max_len - len(tokens))
        
        ## Audio processing
        audio_path = os.path.join(self.root_dir, 'clips', audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        mel_spec = self.mel_spectrogram(waveform)
        
        return torch.tensor(tokens), mel_spec.squeeze(0).t()

def build_vocab(data_dir: str, tsv_files: List[str]) -> Dict[str, int]:
    char_freq = {}
    for tsv_file in tsv_files:
        with open(os.path.join(data_dir, tsv_file), 'r', encoding='utf-8') as f:
            next(f)  ## Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    for char in parts[1]:
                        char_freq[char] = char_freq.get(char, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for char in sorted(char_freq, key=char_freq.get, reverse=True):
        vocab[char] = len(vocab)
    
    return vocab

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts, specs = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, specs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts, specs = [b.to(device) for b in batch]
            
            outputs = model(texts)
            loss = criterion(outputs, specs)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_tts_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_factor'], 
                                  patience=config['scheduler_patience'])
    
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        for key, value in zip(history.keys(), [train_loss, val_loss, current_lr]):
            history[key].append(value)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('models', 'new_tts_model.pth'))
            print("New model saved!")
        
        if epoch - history['val_loss'].index(min(history['val_loss'])) >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {config['early_stopping_patience']} epochs without improvement.")
            break
    
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = 'data/cvc_1'
    tsv_files = ['train.tsv', 'dev.tsv', 'test.tsv']
    
    vocab = build_vocab(data_dir, tsv_files)
    
    train_dataset = CVCDataset(data_dir, 'train.tsv', vocab)
    val_dataset = CVCDataset(data_dir, 'dev.tsv', vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model_config = CVCLionHeart(
        vocab_size=len(vocab),
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        dropout=0.1,
    )
    
    model = CVCModel(model_config).to(device)
    
    train_config = {
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'early_stopping_patience': 10
    }
    
    os.makedirs('models', exist_ok=True)
    
    history = train_tts_model(model, train_loader, val_loader, train_config, device)
    
    import json
    with open(os.path.join('models', 'tts_training_history.json'), 'w') as f:
        json.dump(history, f)
    
    print("Training completed! The new model and training history saved in 'models' directory.")

if __name__ == "__main__":
    main()
