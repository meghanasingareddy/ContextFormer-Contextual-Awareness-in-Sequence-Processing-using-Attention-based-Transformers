import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoderLayerWithAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayerWithAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # get attention weights!
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                          key_padding_mask=memory_key_padding_mask, need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, attn_weights

class ContextFormer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, max_seq_len=100):
        super(ContextFormer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithAttention(d_model, nhead, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx=0):
        return (seq == pad_idx)
        
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # src, tgt shape: (batch_size, seq_len)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        src_seq_len, batch_size = src.shape
        tgt_seq_len, _ = tgt.shape
        
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        src_key_padding_mask = self.create_padding_mask(src.transpose(0,1), src_pad_idx).to(src.device)
        tgt_key_padding_mask = self.create_padding_mask(tgt.transpose(0,1), tgt_pad_idx).to(src.device)
        
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        output = tgt_emb
        attns = []
        for layer in self.decoder_layers:
            output, attn = layer(
                output, memory, 
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            attns.append(attn)
            
        out = self.fc_out(output) # (tgt_seq_len, batch_size, vocab_size)
        out = out.transpose(0, 1) # (batch_size, tgt_seq_len, vocab_size)
        
        return out, attns
