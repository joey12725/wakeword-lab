#!/usr/bin/env python3
"""Train the 'hey Joy' wake word model."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

# OpenWakeWord imports
from openwakeword.model import AudioFeature

class WakeWordFCN(nn.Module):
    """Fully Connected Network for wake word detection."""
    
    def __init__(self, input_size=96, hidden_size=64, num_layers=3):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_audio_files(directory, label):
    """Load all audio files from a directory and assign a label."""
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac']:
        audio_files.extend(Path(directory).rglob(ext))
    
    return [(str(f), label) for f in audio_files]

def generate_embeddings(audio_files_with_labels, output_path, feature_extractor):
    """Generate embeddings for audio files."""
    embeddings = []
    labels = []
    
    print(f"Generating embeddings for {len(audio_files_with_labels)} audio files...")
    
    for audio_path, label in tqdm(audio_files_with_labels):
        try:
            # Load audio and generate embedding
            # OpenWakeWord uses 3-second windows by default
            embedding = feature_extractor.get_features(audio_path)
            
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Save embeddings
    np.savez(output_path, 
             embeddings=np.array(embeddings),
             labels=np.array(labels))
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    return np.array(embeddings), np.array(labels)

def train_model(embeddings, labels, output_dir, epochs=100, batch_size=32):
    """Train the FCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Create model
    input_size = embeddings.shape[1]
    model = WakeWordFCN(input_size=input_size, hidden_size=64, num_layers=3).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining model for up to {epochs} epochs...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate accuracy
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    
    return model

def export_to_onnx(model, output_path, input_size=96):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    print(f"\nModel exported to ONNX: {output_path}")

def main():
    """Main training pipeline."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Initialize feature extractor
    print("Initializing OpenWakeWord feature extractor...")
    feature_extractor = AudioFeature()
    
    # Collect audio files
    print("\nCollecting positive samples...")
    positive_files = []
    positive_files.extend(load_audio_files(data_dir / 'positive_vits', 1))
    positive_files.extend(load_audio_files(data_dir / 'positive_waveglow', 1))
    print(f"Found {len(positive_files)} positive samples")
    
    print("\nCollecting negative samples...")
    negative_files = []
    negative_files.extend(load_audio_files(data_dir / 'adversarial_negatives', 0))
    negative_files.extend(load_audio_files(data_dir / 'negatives', 0))
    print(f"Found {len(negative_files)} negative samples")
    
    # Balance dataset if needed
    all_files = positive_files + negative_files
    print(f"\nTotal samples: {len(all_files)} ({len(positive_files)} positive, {len(negative_files)} negative)")
    
    # Generate embeddings
    embeddings_path = models_dir / 'embeddings.npz'
    if embeddings_path.exists():
        print(f"\nLoading existing embeddings from {embeddings_path}...")
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
    else:
        embeddings, labels = generate_embeddings(all_files, embeddings_path, feature_extractor)
    
    # Train model
    model = train_model(embeddings, labels, models_dir)
    
    # Export to ONNX
    onnx_path = models_dir / 'hey_joy.onnx'
    export_to_onnx(model, onnx_path, input_size=embeddings.shape[1])
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {onnx_path}")

if __name__ == '__main__':
    main()
