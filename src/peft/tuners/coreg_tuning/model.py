# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# from peft.tuners.prompt_tuning import PromptEmbedding
from peft.utils import TaskType

from .config import CoregPromptTuningConfig, CoregPromptTuningInit


# This code is adapted for the paper: 


class CoregPromptEmbedding(torch.nn.Module):
    
    
    def __init__(self, config: CoregPromptTuningConfig):
        super().__init__()

        self.token_dim = config.token_dim
        self.num_views = config.num_views
        self.attention_dim = config.attention_dim       
        self.num_virtual_tokens = config.num_virtual_tokens
        self.decorrelate = config.decorrelate
        self.decorrelate_lambda = config.decorrelate_lambda

        self.num_transformer_submodules = config.num_transformer_submodules
        if self.num_transformer_submodules is None:
            self.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        self.token_dim = config.token_dim
        

        self.total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules 
        
        
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        
        # self.k = torch.nn.Linear(self.attention_dim,self.attention_dim) 
        # self.q = torch.nn.Linear(self.attention_dim,self.attention_dim) 
        # self.v = torch.nn.Linear(self.attention_dim,self.attention_dim) 
        
        
        self.multihead_attn = torch.nn.MultiheadAttention(self.attention_dim, self.num_views, dropout=0.1)
        
        self.mlp_head_k = torch.nn.Linear(self.token_dim, self.attention_dim)
        
        self.mlp_head_q = torch.nn.Linear(self.token_dim, self.attention_dim)
        
        self.mlp_head_v = torch.nn.Linear(self.token_dim, self.attention_dim)
            
        self.proj = torch.nn.Linear(self.attention_dim,self.token_dim) 

    def decorrelate_heads(self, attention_output):
        l=self.decorrelate_lambda
        batch_size, seq_len, _ = attention_output.shape
        num_heads = self.num_views
        head_dim = attention_output.shape[2] // num_heads

        # Reshape the output to separate the heads and combine the batch and sequence length dimensions
        reshaped_output = attention_output.view(batch_size * seq_len, num_heads * head_dim)

        # Center the data
        mean = torch.mean(reshaped_output, dim=0, keepdim=True)
        centered_output = reshaped_output - mean

        # Compute SVD
        U, S, V = torch.linalg.svd(centered_output, full_matrices=False)
        
        # print(centered_output.shape)
        # print(U.shape)
        # print(S.shape)
        # print(V.shape)
        # print(batch_size, seq_len, num_heads, head_dim)

        # Apply the whitening transformation using SVD components
        # We transform the data using U and the inverse square root of S
        decorrelated_output = U.matmul(torch.diag(1.0 / torch.sqrt(S + 1e-5)))

        # Reshape back to the original shape
        decorrelated_output = decorrelated_output.view(batch_size, seq_len, num_heads, head_dim)
        return (1-l)*attention_output + (l)*decorrelated_output.view(batch_size, seq_len, -1)
    
    
    def decorrelate_heads2(self, attention_output):
        l = self.decorrelate_lambda
        batch_size, seq_len, _ = attention_output.shape
        num_heads = self.num_views
        head_dim = attention_output.shape[2] // num_heads

        # Reshape the output to separate the heads and combine the batch and sequence length dimensions
        reshaped_output = attention_output.view(batch_size * seq_len, num_heads * head_dim)

        # Center the data
        mean = torch.mean(reshaped_output, dim=0, keepdim=True)
        centered_output = reshaped_output - mean
        
        
        # Compute SVD
        U, S, V = torch.linalg.svd(centered_output, full_matrices=False)

        # Apply the whitening transformation using SVD components
        singular_value_matrix = torch.diag(1.0 / torch.sqrt(S + 1e-5))
        reduced_output = centered_output.matmul(V.transpose(0, 1))
        scaled_output = reduced_output.matmul(singular_value_matrix)
        whitened_output = scaled_output.matmul(V)

        # Reshape back to the original shape
        whitened_output = whitened_output.view(batch_size, seq_len, num_heads, head_dim)
        return (1 - l) * attention_output + l * whitened_output.view(batch_size, seq_len, -1)






    def forward(self, indices):

        prompt_embeddings = self.embedding(indices)
        
        k = self.mlp_head_k(prompt_embeddings)
        q = self.mlp_head_q(prompt_embeddings)
        v = self.mlp_head_v(prompt_embeddings)
        
        # k = self.k(ek)
        # q = self.q(eq)
        # v = self.v(ev)
        
        consolidated_view,_ = self.multihead_attn(q,k,v)
    

        # Apply decorrelation
        if self.decorrelate:
            consolidated_view = self.decorrelate_heads2(consolidated_view)

        return self.proj(consolidated_view)
    