import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, embed_dim, hidded_dim, num_enbed, num_pos, num_heads, num_layers, dropout):
        super().__init__()
        self.tokens_embeddings = nn.Embedding(num_enbed, embed_dim)
        self.position_embeddings = nn.Embedding(num_pos, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.attention, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.ln_1, self.ln_2 = nn.ModuleList(), nn.ModuleList()

        for _ in range(num_layers):
            self.attention.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidded_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidded_dim, embed_dim)))

            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    
    
    def forward(self, x):
        position = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(position).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.ln_1, self.attention,
                                                                       self.ln_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h

        return h.contiguous().view(-1, self.embed_dim)
