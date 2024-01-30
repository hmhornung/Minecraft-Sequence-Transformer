import numpy as np
import torch
import torch.nn as nn

class PositionalEmbedding3D(nn.Module):
    def __init__(self, d_model, src_shape, tgt_shape, tgt_offset):
        super(PositionalEmbedding3D, self).__init__()
        assert d_model % 3 == 0, 'd_model must be divisible by 3'
        
        self.x_max_seq_len = tgt_offset[0] + tgt_shape[0]
        self.y_max_seq_len = tgt_offset[1] + tgt_shape[1]
        self.z_max_seq_len = tgt_offset[2] + tgt_shape[2]
        
        self.pos_embedding_x = nn.Embedding(self.x_max_seq_len + 1 , d_model // 3) #plus 1 for the SOS number
        self.pos_embedding_y = nn.Embedding(self.y_max_seq_len + 1 , d_model // 3)
        self.pos_embedding_z = nn.Embedding(self.z_max_seq_len + 1, d_model // 3)
        
        self.src_positions = np.empty(src_shape, dtype=object)
        self.tgt_positions = np.empty(tgt_shape, dtype=object)
        
        # -----------------------------------------------
        # Create 3-D array corresponding to 3-D coordinates
        # for each token in the src & tgt sequence
        # -----------------------------------------------
        for i in range(src_shape[0]):
            for j in range(src_shape[1]):
                for k in range(src_shape[2]):
                    self.src_positions[i, j, k] = (i, j, k)
        
        for i in range(tgt_shape[0]):
            for j in range(tgt_shape[1]):
                for k in range(tgt_shape[2]):
                    self.tgt_positions[i, j, k] = (i + tgt_offset[0], j + tgt_offset[1], k + tgt_offset[2])
        
        
        # flatten the arrays to be used for the 1-D sequences
        self.src_positions = self.src_positions.ravel() 
        self.tgt_positions = self.tgt_positions.ravel() 
        print(self.tgt_positions)
        # ------------------------------------------------------------
        # Separate the 3 dimensions from the tuple arrays into tensors
        # ------------------------------------------------------------
        self.src_pos_x = torch.tensor([i[0] for i in self.src_positions])
        self.src_pos_y = torch.tensor([i[1] for i in self.src_positions])
        self.src_pos_z = torch.tensor([i[2] for i in self.src_positions])

        self.tgt_pos_x = [i[0] for _ , i in np.ndenumerate(self.tgt_positions.flat)]
        self.tgt_pos_y = [i[1] for _ , i in np.ndenumerate(self.tgt_positions.flat)]
        self.tgt_pos_z = [i[2] for _ , i in np.ndenumerate(self.tgt_positions.flat)]
        print(f"{type(self.tgt_pos_x)} {type(self.tgt_pos_y)} {type(self.tgt_pos_z)} ")
        self.tgt_pos_x.insert(0, self.x_max_seq_len)
        self.tgt_pos_y.insert(0, self.y_max_seq_len)
        self.tgt_pos_z.insert(0, self.z_max_seq_len)
        self.tgt_pos_x = torch.tensor(self.tgt_pos_x) # Insert the SOS token at beginning of
        self.tgt_pos_y = torch.tensor(self.tgt_pos_y) # tgt flattened token to position arrays
        self.tgt_pos_z = torch.tensor(self.tgt_pos_z) # --------------------------------------
        
        print(f"{self.tgt_pos_x}\n{self.tgt_pos_y}\n{self.tgt_pos_z} ")
        
    def forward(self, x, src_tgt: bool): # Batch, Seq, D_Model
        """
        src_tgt: 
        TRUE->src 
        FALSE->tgt
        """
        if src_tgt:
            pos_embedding = torch.cat([self.pos_embedding_x(self.src_pos_x), self.pos_embedding_y(self.src_pos_y), self.pos_embedding_z(self.src_pos_z)], dim=-1)
        else:
            pos_embedding = torch.cat([self.pos_embedding_x(self.tgt_pos_x[:x.size(1)]), self.pos_embedding_y(self.tgt_pos_y[:x.size(1)]), self.pos_embedding_z(self.tgt_pos_z[:x.size(1)])], dim=-1)
        print(pos_embedding.shape)
        return x + pos_embedding