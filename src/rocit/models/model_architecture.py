import torch
from torch import nn
class SparseEmbeddingBlock(nn.Module):
    def __init__(self, dim:int,missing_idx:int=0):
        super().__init__()
        self.dim = dim
        self.missing_idx = missing_idx
        print(self.dim)
        self.impute_values = nn.Parameter(torch.zeros((dim,)))
        self.missing_vector = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        # Placeholders for runtime context 
        self.register_buffer('_embedding', None, persistent=False)
        self.register_buffer('_nan_mask', None, persistent=False)
    
    def set_context(self, embedding_source):
        """Call once per dataset before any forward passes."""
        embedding = embedding_source.get_embedding_vector().to(self.impute_values.device)
        

        nan_mask = torch.isnan(embedding)
        embedding_clean = embedding.clone()
        embedding_clean[nan_mask] = 0.0
        
        self._embedding = embedding_clean
        self._nan_mask = nan_mask if nan_mask.any() else None

    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            raise RuntimeError("Must call set_context() before forward pass")
        is_missing_entity = (idx == self.missing_idx).unsqueeze(-1)
        
        #shift them down again
        vectors = self._embedding[idx-1]
        
        if self._nan_mask is not None:
            feature_nan_mask = self._nan_mask[idx-1]
            
            vectors = torch.where(feature_nan_mask, self.impute_values, vectors)
        
        return torch.where(is_missing_entity, self.missing_vector, vectors)



class ROCITClassifier(nn.Module):

    SCALE_CONSTANT:float = 0.05
    def __init__(self, emb, n_heads, n_blocks,seq_length=511,dropout_rate=0.2,sample_distribution_dim=19,cell_map_dim=84):
        super().__init__()

        self.sample_distribution_dim = sample_distribution_dim
        self.cell_map_dim = cell_map_dim
        self.sample_distribution_embedding = SparseEmbeddingBlock(self.sample_distribution_dim)
        self.cell_map_embedding = SparseEmbeddingBlock(self.cell_map_dim)

        self.emb = emb
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.pos_emb = nn.Embedding(seq_length, emb)
        self.dropout=dropout_rate


        self.class_vector = nn.Parameter(torch.randn(emb)*self.SCALE_CONSTANT)
        self.cell_type_embedder  = nn.Sequential(
            nn.Linear(in_features=self.cell_map_dim, out_features=emb),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(in_features=emb, out_features=emb),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(in_features=emb, out_features=emb)
        )

        #add two to input dimension for position and methylation probability
        self.methylation_embedder  = nn.Sequential(
            nn.Linear(in_features=self.sample_distribution_dim+2, out_features=emb),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(in_features=emb, out_features=emb),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(in_features=emb, out_features=emb),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=n_heads,dim_feedforward=4*emb, batch_first=True,bias=False,norm_first=True,activation='gelu',dropout=self.dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        self.to_output_probability= nn.Sequential(nn.Dropout(self.dropout),nn.Linear(in_features=emb, out_features=1))

    def set_embedding_context(self, embedding_sources: dict):
        self.sample_distribution_embedding.set_context(embedding_sources['sample_distribution'])
        self.cell_map_embedding.set_context(embedding_sources['cell_map'])
        

    def forward(self, methylation,read_position,sample_distribution_index,cell_map_index,attention_mask,**kwargs):

        cell_type_methylation_sample = self.cell_map_embedding(cell_map_index)
        
        position_methylation_sample= self.sample_distribution_embedding(sample_distribution_index)
        
        
        input_vector = self.cell_type_embedder(cell_type_methylation_sample) + self.methylation_embedder(torch.cat([position_methylation_sample,methylation.unsqueeze(-1),read_position.unsqueeze(-1)],dim=-1))

        pos_emb = self.pos_emb(torch.arange(input_vector.shape[1], device=input_vector.device))[None, :, :].expand_as(input_vector)
        
        input_vector = input_vector + pos_emb*self.SCALE_CONSTANT
     
        class_emb = self.class_vector.view(1,1,-1).expand(input_vector.shape[0],-1,-1)

        input_vector = torch.cat([class_emb,input_vector],dim=1)

        x= self.transformer_encoder(input_vector,src_key_padding_mask=attention_mask.bool())
        out= x[:,0].reshape(x.shape[0],-1)
   
       
        class_probs = self.to_output_probability(out).view(-1)
    
        return class_probs