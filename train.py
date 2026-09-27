import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import ContextFormer
import os

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, (src, tgt) in enumerate(iterator):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output, _ = model(src, tgt_input)
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _ = model(src, tgt_input)
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = 30
    batch_size = 128
    
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, vocab_size=vocab_size)
    
    model = ContextFormer(vocab_size=vocab_size, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    epochs = 15
    clip = 1.0
    
    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
        
    torch.save(model.state_dict(), 'context_former.pt')
    print("Model saved to context_former.pt")

if __name__ == '__main__':
    main()
