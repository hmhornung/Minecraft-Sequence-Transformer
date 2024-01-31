import torch
import torch.nn as nn
from PositionalEmbedding3D import PositionalEmbedding3D
# from PositionalEmbedding3D import PositionalEmbedding3D

class MinecraftSequencePredict(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, src_shape: tuple, tgt_shape: tuple, tgt_offset: tuple, device):
        super(MinecraftSequencePredict, self).__init__()

        # Define the embedding layer
        self.block_embedding = nn.Embedding(vocab_size, d_model, device=device)

        self.pos_embedding = PositionalEmbedding3D(d_model, src_shape, tgt_shape, tgt_offset, device=device)
        # Define the transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            device=device
        )

        # Define the output layer
        self.fc = nn.Linear(d_model, vocab_size,device=device)
        self.softmax = nn.Softmax(dim=-1)
        self.verbose = False

    def forward(self, src, tgt):
        if self.verbose: print(f'sizes:\nsrc: {src.shape}\ntgt: {tgt.shape}\n')
        
        src = self.block_embedding(src)
        tgt = self.block_embedding(tgt)
        
        if self.verbose: print(f'sizes after block embeddings:\nsrc: {src.shape}\ntgt: {tgt.shape}\n')
        
        src = self.pos_embedding(src, True)
        tgt = self.pos_embedding(tgt, False)
        
        if self.verbose: print(f'sizes after positional embeddings:\nsrc: {src.shape}\ntgt: {tgt.shape}\n')
        
        output = self.transformer(src, tgt)
        
        if self.verbose: print(f'output shape from transformer:\n{output.shape}\n')

        output = self.fc(output)
        
        if self.verbose: print(f'output shape after linear layer:\n{output.shape}\n')
        
        output = self.softmax(output)

        return output
    def verbose(self, verbose: bool = True):
        self.verbose = verbose