import numpy as np
import torch
import torch.nn as nn

class PositionalEmbedding3D(nn.Module):
    def __init__(self, d_model, src_shape, tgt_shape, tgt_offset, device):
        super(PositionalEmbedding3D, self).__init__()
        assert d_model % 3 == 0, 'd_model must be divisible by 3'
        
        self.d_model = d_model
        self.device = device
        # -----------------------------------------------------
        # Create max for (x,y,z) that will be used as the 
        # vocab length used in the positional embedding modules
        # -----------------------------------------------------
        self.x_max_seq_len = tgt_offset[0] + tgt_shape[0]
        self.y_max_seq_len = tgt_offset[1] + tgt_shape[1]
        self.z_max_seq_len = tgt_offset[2] + tgt_shape[2]
        
        self.pos_embedding_x = nn.Embedding(self.x_max_seq_len, d_model // 3).to(device)
        self.pos_embedding_y = nn.Embedding(self.y_max_seq_len, d_model // 3).to(device)
        self.pos_embedding_z = nn.Embedding(self.z_max_seq_len, d_model // 3).to(device)
        
        # -------------------------------------------------
        # Create 3-D array corresponding to 3-D coordinates
        # for each token in the src & tgt sequence
        # -------------------------------------------------
        self.src_positions = np.empty(src_shape, dtype=object)
        self.tgt_positions = np.empty(tgt_shape, dtype=object)
        
        for i in range(src_shape[0]):
            for j in range(src_shape[1]):
                for k in range(src_shape[2]):
                    self.src_positions[i, j, k] = (i, j, k)
        
        for i in range(tgt_shape[0]):
            for j in range(tgt_shape[1]):
                for k in range(tgt_shape[2]):
                    #make the tgt positons come after the src with the tgt_offset
                    self.tgt_positions[i, j, k] = (i + tgt_offset[0], j + tgt_offset[1], k + tgt_offset[2])
        
        
        # flatten the arrays to be used for the 1-D sequences
        self.src_positions = self.src_positions.reshape(-1) 
        self.tgt_positions = self.tgt_positions.reshape(-1) 

        # ------------------------------------------------------------
        # Separate the 3 dimensions from the tuple arrays into tensors
        # ------------------------------------------------------------
        self.src_pos_x = torch.tensor([i[0] for i in self.src_positions], device=device)
        self.src_pos_y = torch.tensor([i[1] for i in self.src_positions], device=device)
        self.src_pos_z = torch.tensor([i[2] for i in self.src_positions], device=device)
        
        self.tgt_pos_x = torch.tensor([i[0] for i in self.tgt_positions], device=device)
        self.tgt_pos_y = torch.tensor([i[1] for i in self.tgt_positions], device=device)
        self.tgt_pos_z = torch.tensor([i[2] for i in self.tgt_positions], device=device)
        
    def forward(self, x, src_tgt: bool): # Batch, Seq, D_Model
        """
        src_tgt: 
        TRUE->src 
        FALSE->tgt
        """
        
        # print(x.device)
        # print(self.pos_embedding_x.weight.device)
        # print(self.src_pos_x.device)
        
        if src_tgt:
            self.pos_embedding_x(self.src_pos_x)
            pos_embedding = torch.cat([self.pos_embedding_x(self.src_pos_x), self.pos_embedding_y(self.src_pos_y), self.pos_embedding_z(self.src_pos_z)], dim=-1)
        else:
            pos_embedding = torch.cat([self.pos_embedding_x(self.tgt_pos_x[:x.size(-1) - 1]), self.pos_embedding_y(self.tgt_pos_y[:x.size(1) - 1]), self.pos_embedding_z(self.tgt_pos_z[:x.size(1) - 1])], dim=-1)
            # (1 x d_model) + (seq_len x d_model)
            pos_embedding = torch.cat([torch.zeros(( 1 , self.d_model ), device=self.device), pos_embedding], dim=0)
            # -- the SOS token will have no positional embedding -- #
        
        return x + pos_embedding