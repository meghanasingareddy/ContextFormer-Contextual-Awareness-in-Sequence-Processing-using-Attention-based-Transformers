import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ContextFormer
import os

def generate_sequence_and_get_attention(model, src_seq, device, max_len=20, sos_idx=1, eos_idx=2):
    model.eval()
    
    src_tensor = torch.tensor(src_seq).unsqueeze(0).to(device)
    tgt_indices = [sos_idx]
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            output, attns = model(src_tensor, tgt_tensor)
            
            pred_token = output.argmax(2)[:,-1].item()
            tgt_indices.append(pred_token)
            
            if pred_token == eos_idx:
                break
                
    # Extract attention from the last layer. 
    # attns[-1] has shape (batch_size, tgt_len, src_len)
    final_attention = attns[-1].squeeze(0).cpu().numpy()
                
    return tgt_indices[1:-1], final_attention

def plot_attention(src, tgt, attention, filename="attention.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(src)))
    ax.set_yticks(np.arange(len(tgt)))
    
    ax.set_xticklabels([str(x) for x in src])
    ax.set_yticklabels([str(x) for x in tgt])

    plt.xlabel("Source Sequence")
    plt.ylabel("Target Sequence (incl. SOS)")
    plt.title("Encoder-Decoder Attention Weights")
    plt.savefig(filename)
    print(f"Attention plot saved to {filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 30
    
    model = ContextFormer(vocab_size=vocab_size, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2).to(device)
    
    if os.path.exists('context_former.pt'):
        model.load_state_dict(torch.load('context_former.pt', map_location=device))
        print("Loaded model from context_former.pt")
    else:
        print("Model file not found. Please train first.")
        return
        
    src_seq = [10, 15, 20, 25, 5, 8] # Random example sequence
    print(f"Input sequence: {src_seq}")
    print(f"Expected output: {src_seq[::-1]}")
    
    tgt_seq, attention = generate_sequence_and_get_attention(model, src_seq, device)
    
    print(f"Predicted output: {tgt_seq}")
    
    # tgt sequence has SOS token at the start in the attention matrix.
    tgt_labels = ['<SOS>'] + [str(x) for x in tgt_seq] + (['<EOS>'] if len(tgt_seq) < attention.shape[0]-1 else [])
    # Pad tgt_labels if necessary based on attention shape, or slice attention
    attention = attention[:len(tgt_labels), :len(src_seq)]
    
    plot_attention(src_seq, tgt_labels, attention, filename="attention.png")

if __name__ == '__main__':
    main()
