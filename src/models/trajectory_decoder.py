"""Trajectory Decoder: pre-norm transformer decoder reading the scene memory."""

import torch.nn as nn


class TrajectoryDecoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8, num_layers=4, ffn_dim=512,
                 dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, num_heads, ffn_dim, dropout=dropout,
                batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, S_t, scene_memory, tgt_mask=None):
        """S_t [B,H,C], scene_memory [B,Ns,C] -> [B,H,C]."""
        out = S_t
        for layer in self.layers:
            out = layer(out, scene_memory, tgt_mask=tgt_mask)
        return out
