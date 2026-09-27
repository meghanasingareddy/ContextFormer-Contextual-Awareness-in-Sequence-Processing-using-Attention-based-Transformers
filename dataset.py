import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StringReversalDataset(Dataset):
    def __init__(self, num_samples, max_seq_len, vocab_size):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Vocab: 0 is PAD, 1 is SOS, 2 is EOS, 3+ are characters
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        
        self.data = self._generate_data()
        
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # random sequence length
            seq_len = np.random.randint(1, self.max_seq_len)
            
            # random character sequence (integers from 3 to vocab_size-1)
            src = np.random.randint(3, self.vocab_size, size=seq_len).tolist()
            
            # target is reversed source
            tgt = src[::-1]
            
            # Add SOS and EOS tokens to target
            tgt = [self.SOS_IDX] + tgt + [self.EOS_IDX]
            
            # Pad sequences (Source has max length max_seq_len, Target has max length max_seq_len + 2)
            src_padded = src + [self.PAD_IDX] * (self.max_seq_len - len(src))
            tgt_padded = tgt + [self.PAD_IDX] * (self.max_seq_len + 2 - len(tgt))
            
            data.append((torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloaders(batch_size=32, num_samples=10000, max_seq_len=15, vocab_size=30):
    dataset = StringReversalDataset(num_samples, max_seq_len, vocab_size)
    
    # Split into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=2)
    src, tgt = next(iter(train_loader))
    print("Source shape:", src.shape)
    print("Source:\n", src)
    print("Target shape:", tgt.shape)
    print("Target:\n", tgt)
