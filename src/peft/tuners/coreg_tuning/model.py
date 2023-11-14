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

       
    def forward(self, indices):

        prompt_embeddings = self.embedding(indices)
        
        k = self.mlp_head_k(prompt_embeddings)
        q = self.mlp_head_q(prompt_embeddings)
        v = self.mlp_head_v(prompt_embeddings)
        
        # k = self.k(ek)
        # q = self.q(eq)
        # v = self.v(ev)
        
        consolidated_view,_ = self.multihead_attn(q,k,v)

        return self.proj(consolidated_view)
    