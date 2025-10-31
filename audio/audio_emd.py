#!/usr/bin/env python3
# conformer_audio_classifier.py
# Conformer-based Audio Embedding + MLP Classifier với lưu trữ riêng biệt

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
from typing import Optional, Tuple, Dict, List

# ==========================================
# CONFORMER ENCODER
# ==========================================
class ConformerBlock(nn.Module):
    """
    Single Conformer block: FFN -> Multi-Head Self-Attention -> Conv -> FFN
    """
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed-forward module 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)
        
        # Convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_final = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        
        # FFN 1 (half residual)
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head self-attention
        normed = self.norm_attn(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout_attn(attn_out)
        
        # Convolution module
        normed = self.norm_conv(x)
        conv_in = normed.transpose(1, 2)  # (batch, d_model, seq_len)
        conv_out = self.conv(conv_in).transpose(1, 2)
        x = x + conv_out
        
        # FFN 2 (half residual)
        x = x + 0.5 * self.ffn2(x)
        
        return self.norm_final(x)


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder cho audio embedding
    """
    def __init__(
        self, 
        input_dim=80,          # Mel-spectrogram features
        d_model=256,           # Hidden dimension
        num_layers=4,          # Number of Conformer blocks
        num_heads=4,           # Attention heads
        conv_kernel_size=31,   # Convolution kernel size
        output_dim=128,        # Final embedding dimension
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim) hoặc (batch, input_dim) nếu đã pooled
            lengths: (batch,) độ dài thực của mỗi sequence
        Returns:
            embeddings: (batch, output_dim)
        """
        # Nếu input đã là vector (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Temporal pooling (mean over time)
        if lengths is not None:
            # Masked mean pooling
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            x = (x * mask).sum(1) / mask.sum(1)
        else:
            x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        embeddings = self.output_proj(x)  # (batch, output_dim)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# ==========================================
# MLP CLASSIFIER
# ==========================================
class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier
    """
    def __init__(
        self, 
        input_dim=128,
        hidden_dims=[256, 128, 64],
        num_classes=10,
        dropout=0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


# ==========================================
# CONFORMER TRAINER
# ==========================================
class ConformerTrainer:
    """
    Trainer cho Conformer encoder
    """
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder.to(device)
        self.device = device
    
    def train_contrastive(
        self, 
        X_train, 
        X_val, 
        epochs=100,
        batch_size=32,
        lr=1e-3,
        temperature=0.07
    ):
        """
        Train với contrastive learning (SimCLR-style)
        """
        print("\n🔥 Training Conformer with Contrastive Learning...")
        
        train_dataset = torch.FloatTensor(X_train)
        val_dataset = torch.FloatTensor(X_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Train
            self.encoder.train()
            train_loss = 0
            
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                
                # Create augmented views (simple: add noise)
                noise1 = torch.randn_like(batch_x) * 0.1
                noise2 = torch.randn_like(batch_x) * 0.1
                x1 = batch_x + noise1
                x2 = batch_x + noise2
                
                # Get embeddings
                z1 = self.encoder(x1)
                z2 = self.encoder(x2)
                
                # NT-Xent loss
                loss = self.nt_xent_loss(z1, z2, temperature)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.encoder.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x in val_loader:
                    batch_x = batch_x.to(self.device)
                    
                    noise1 = torch.randn_like(batch_x) * 0.1
                    noise2 = torch.randn_like(batch_x) * 0.1
                    x1 = batch_x + noise1
                    x2 = batch_x + noise2
                    
                    z1 = self.encoder(x1)
                    z2 = self.encoder(x2)
                    
                    loss = self.nt_xent_loss(z1, z2, temperature)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.encoder.state_dict(), 'conformer_encoder_best.pt')
        
        print(f"✅ Training completed! Best val loss: {best_val_loss:.4f}")
        return history
    
    def nt_xent_loss(self, z1, z2, temperature=0.07):
        """
        Normalized Temperature-scaled Cross Entropy Loss
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
        
        # Cosine similarity
        sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)
        
        # Remove diagonal
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
        
        # Mask to remove self-similarity
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool, device=z.device)
        mask.fill_diagonal_(False)
        
        negative_samples = sim[mask].reshape(2 * batch_size, -1)
        
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


# ==========================================
# MLP CLASSIFIER TRAINER
# ==========================================
class MLPTrainer:
    """
    Trainer cho MLP classifier
    """
    def __init__(self, classifier, device='cuda'):
        self.classifier = classifier.to(device)
        self.device = device
    
    def train(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64,
        lr=1e-3
    ):
        """
        Train MLP classifier
        """
        print("\n🎯 Training MLP Classifier...")
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Train
            self.classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.classifier(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, pred = logits.max(1)
                train_total += batch_y.size(0)
                train_correct += pred.eq(batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validate
            self.classifier.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.classifier(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    _, pred = logits.max(1)
                    val_total += batch_y.size(0)
                    val_correct += pred.eq(batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.classifier.state_dict(), 'mlp_classifier_best.pt')
        
        print(f"✅ Training completed! Best val accuracy: {best_val_acc:.3f}")
        return history


# ==========================================
# SAVE/LOAD FUNCTIONS
# ==========================================

def save_conformer_encoder(encoder, filename='conformer_encoder.pkl'):
    """
    Lưu Conformer encoder
    """
    model_data = {
        'state_dict': encoder.state_dict(),
        'config': {
            'input_dim': encoder.input_dim,
            'd_model': encoder.d_model,
            'output_dim': encoder.output_dim,
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"✅ Conformer encoder saved to {filename} ({size_mb:.2f} MB)")


def load_conformer_encoder(filename='conformer_encoder.pkl', device='cpu'):
    """
    Load Conformer encoder
    """
    print(f"📂 Loading Conformer encoder from {filename}...")
    
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    config = model_data['config']
    encoder = ConformerEncoder(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        output_dim=config['output_dim']
    )
    
    encoder.load_state_dict(model_data['state_dict'])
    encoder.to(device)
    encoder.eval()
    
    print(f"✅ Encoder loaded (input_dim={config['input_dim']}, output_dim={config['output_dim']})")
    return encoder


def save_mlp_classifier(classifier, filename='mlp_classifier.pkl'):
    """
    Lưu MLP classifier
    """
    model_data = {
        'state_dict': classifier.state_dict(),
        'config': {
            'input_dim': classifier.input_dim,
            'num_classes': classifier.num_classes,
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"✅ MLP classifier saved to {filename} ({size_mb:.2f} MB)")


def load_mlp_classifier(filename='mlp_classifier.pkl', device='cpu'):
    """
    Load MLP classifier
    """
    print(f"📂 Loading MLP classifier from {filename}...")
    
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    config = model_data['config']
    classifier = MLPClassifier(
        input_dim=config['input_dim'],
        num_classes=config['num_classes']
    )
    
    classifier.load_state_dict(model_data['state_dict'])
    classifier.to(device)
    classifier.eval()
    
    print(f"✅ Classifier loaded (input_dim={config['input_dim']}, num_classes={config['num_classes']})")
    return classifier


def save_audio_embeddings(embeddings, labels, metadata, filename='audio_embeddings.pkl'):
    """
    Lưu audio embeddings (raw features từ audio)
    """
    data = {
        'embeddings': embeddings,  # (N, D)
        'labels': labels,          # (N,)
        'metadata': metadata,      # List[Dict]
        'shape': embeddings.shape
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"✅ Audio embeddings saved to {filename}")
    print(f"   Shape: {embeddings.shape}, Size: {size_mb:.2f} MB")


def load_audio_embeddings(filename='audio_embeddings.pkl'):
    """
    Load audio embeddings
    """
    print(f"📂 Loading audio embeddings from {filename}...")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded embeddings shape: {data['shape']}")
    return data['embeddings'], data['labels'], data['metadata']


def extract_embeddings_batch(encoder, X, batch_size=64, device='cuda'):
    """
    Extract embeddings từ Conformer encoder
    """
    encoder.eval()
    encoder.to(device)
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Extracting embeddings"):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            z = encoder(batch)
            embeddings.append(z.cpu().numpy())
    
    return np.vstack(embeddings)


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎵 CONFORMER AUDIO ENCODER + MLP CLASSIFIER")
    print("=" * 60)
    
    # Example: Tạo dummy data
    print("\n📊 Creating dummy data...")
    n_samples = 1000
    input_dim = 80  # Mel-spectrogram features
    num_classes = 10
    
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # 1. Train Conformer Encoder
    print("\n" + "=" * 60)
    print("STEP 1: Train Conformer Encoder")
    print("=" * 60)
    
    encoder = ConformerEncoder(input_dim=input_dim, output_dim=128)
    trainer = ConformerTrainer(encoder, device='cpu')
    
    # Train (sử dụng epochs nhỏ cho demo)
    history = trainer.train_contrastive(X_train, X_val, epochs=20, batch_size=32)
    
    # Save encoder
    save_conformer_encoder(encoder, 'conformer_encoder.pkl')
    
    # 2. Extract embeddings
    print("\n" + "=" * 60)
    print("STEP 2: Extract Embeddings")
    print("=" * 60)
    
    encoder.eval()
    with torch.no_grad():
        train_embeddings = encoder(torch.FloatTensor(X_train)).numpy()
        val_embeddings = encoder(torch.FloatTensor(X_val)).numpy()
    
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Val embeddings: {val_embeddings.shape}")
    
    # Save embeddings
    metadata = [{'idx': i} for i in range(len(X))]
    save_audio_embeddings(X, y, metadata, 'audio_embeddings.pkl')
    
    # 3. Train MLP Classifier
    print("\n" + "=" * 60)
    print("STEP 3: Train MLP Classifier")
    print("=" * 60)
    
    classifier = MLPClassifier(input_dim=128, num_classes=num_classes)
    mlp_trainer = MLPTrainer(classifier, device='cpu')
    
    history = mlp_trainer.train(
        train_embeddings, y_train,
        val_embeddings, y_val,
        epochs=30, batch_size=32
    )
    
    # Save classifier
    save_mlp_classifier(classifier, 'mlp_classifier.pkl')
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("=" * 60)
    print("\n📁 Files created:")
    print("   - conformer_encoder.pkl (Conformer encoder)")
    print("   - mlp_classifier.pkl (MLP classifier)")
    print("   - audio_embeddings.pkl (Raw audio embeddings)")
    print("\n🚀 Ready for inference!")