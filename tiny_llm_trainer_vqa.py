import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from typing import Dict, Tuple, List
import json
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VQALionHeart:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class VQAModel(nn.Module):
    def __init__(self, config: VQALionHeart):
        super().__init__()
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.embed_dim)
        
        self.question_encoder = nn.Embedding(config.vocab_size, config.embed_dim)
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.embed_dim, nhead=config.num_heads),
            num_layers=config.num_layers
        )
        
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)
        
    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        img_features = self.image_encoder(image).unsqueeze(1)  ## (batch_size, 1, embed_dim)
        que_features = self.question_encoder(question)  ## (batch_size, seq_len, embed_dim)
        
        ## Create a causal mask for the transformer
        seq_len = question.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=question.device) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        
        output = self.transformer(que_features.transpose(0, 1), img_features.transpose(0, 1), tgt_mask=causal_mask)
        return self.output_layer(output.transpose(0, 1))

class FlickrVQADataset(Dataset):
    def __init__(self, root_dir: str, vocab: Dict[str, int], max_seq_length: int = 20, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'Images')
        self.captions_file = os.path.join(root_dir, 'captions.txt')
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.data = self.load_captions()
        
    def load_captions(self):
        data = []
        with open(self.captions_file, 'r') as f:
            for line in f:
                img_file, caption = line.strip().split(',', 1)
                data.append((img_file, caption))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, seq: List[int]) -> List[int]:
        return seq[:self.max_seq_length] + [self.vocab['<PAD>']] * (self.max_seq_length - len(seq))
    
    def __getitem__(self, idx):
        img_file, caption = self.data[idx]
        image_path = os.path.join(self.images_dir, img_file)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in caption.split()]
        tokens = [self.vocab['<START>']] + tokens + [self.vocab['<END>']]
        
        input_seq = self.pad_sequence(tokens[:-1])
        target_seq = self.pad_sequence(tokens[1:])
        
        return image, torch.tensor(input_seq), torch.tensor(target_seq)

def build_vocab(datasets: List[str], min_freq: int = 5) -> Dict[str, int]:
    word_freq = {}
    for dataset_path in datasets:
        captions_file = os.path.join(dataset_path, 'captions.txt')
        with open(captions_file, 'r') as f:
            for line in f:
                _, caption = line.strip().split(',', 1)
                for word in caption.split():
                    word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images, questions, targets = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 2)
        total += targets.numel()
        correct += (predicted == targets).sum().item()
    
    return total_loss / len(dataloader), correct / total

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, questions, targets = [b.to(device) for b in batch]
            
            outputs = model(images, questions)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 2)
            total += targets.numel()
            correct += (predicted == targets).sum().item()
    
    return total_loss / len(dataloader), correct / total

def train_vqa_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                    config: Dict, device: torch.device) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss(ignore_index=0)  ## ignore padding index
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=config['scheduler_factor'], 
                                  patience=config['scheduler_patience'])
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        for key, value in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc, current_lr]):
            history[key].append(value)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join('models', 'best_vqa_model.pth'))
            print("New best model saved!")
        
        ## Corrected early stopping condition
        if epoch - max(range(len(history['val_acc'])), key=lambda i: history['val_acc'][i]) >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {config['early_stopping_patience']} epochs without improvement.")
            break
    
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Build vocabulary
    dataset_paths = ['data/flickr8k', 'data/flickr30k']
    vocab = build_vocab(dataset_paths)

    ## Load datasets
    max_seq_length = 20  # Choose an appropriate maximum sequence length
    flickr8k_dataset = FlickrVQADataset('data/flickr8k', vocab, max_seq_length)
    flickr30k_dataset = FlickrVQADataset('data/flickr30k', vocab, max_seq_length)
    
    ## Combine datasets
    combined_dataset = ConcatDataset([flickr8k_dataset, flickr30k_dataset])
    
    ## Split into train and validation
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    ## Model configuration
    model_config = VQALionHeart(
        vocab_size=len(vocab),
        embed_dim=512,
        num_heads=8,
        num_layers=6,
    )
    
    model = VQAModel(model_config).to(device)
    
    ## Training configuration
    config = {
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'scheduler_factor': 0.1,
        'scheduler_patience': 2,
        'early_stopping_patience': 5
    }
    
    ## Create 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    ## Train the model
    history = train_vqa_model(model, train_loader, val_loader, config, device)
    
    ## Save training history
    with open(os.path.join('models', 'vqa_training_history.json'), 'w') as f:
        json.dump(history, f)
    
    print("Training completed. Best model and training history saved in 'models' directory.")

if __name__ == "__main__":
    main()