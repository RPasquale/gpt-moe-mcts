import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from .configuration_gpt_moe_mcts import GPTMoEMCTSConfig

class FlashAttention3(nn.Module):
    def __init__(self, d_model, n_heads, block_size_q, block_size_kv, num_blocks_kv, device='cuda'):
        super(FlashAttention3, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.num_blocks_kv = num_blocks_kv
        self.device = device

        self.q_proj = nn.Linear(d_model, d_model).to(device)
        self.k_proj = nn.Linear(d_model, d_model).to(device)
        self.v_proj = nn.Linear(d_model, d_model).to(device)
        self.out_proj = nn.Linear(d_model, d_model).to(device)

    def forward(self, x):
        B, T, C = x.size()
        Q = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        O = torch.zeros(B, self.n_heads, T, C // self.n_heads).to(self.device)
        L = torch.zeros(B, self.n_heads, T).to(self.device)
        M = torch.full((B, self.n_heads, T), -float('inf')).to(self.device)

        for i in range(0, T, self.block_size_q):
            Q_block = Q[:, :, i:i+self.block_size_q]
            O_block = torch.zeros_like(Q_block).to(self.device)
            L_block = torch.zeros(B, self.n_heads, Q_block.size(2)).to(self.device)
            M_block = torch.full((B, self.n_heads, Q_block.size(2)), -float('inf')).to(self.device)

            for j in range(0, T, self.block_size_kv):
                K_block = K[:, :, j:j+self.block_size_kv]
                V_block = V[:, :, j:j+self.block_size_kv]

                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1))
                M_block_old = M_block
                M_block = torch.max(M_block, S_block.max(dim=-1).values)

                exp_S_block = torch.exp(S_block - M_block.unsqueeze(-1))
                L_block = torch.exp(M_block_old - M_block) * L_block + exp_S_block.sum(dim=-1)

                O_block += torch.matmul(exp_S_block, V_block)

            O_block /= L_block.unsqueeze(-1)
            O[:, :, i:i+self.block_size_q] = O_block

        O = O.transpose(1, 2).contiguous().view(B, T, self.n_heads * (C // self.n_heads))
        O = self.out_proj(O)

        return O

# Define the MLP module
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.scale_init = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Define the MixtureOfExperts module
class MixtureOfExperts(nn.Module):
    def __init__(self, config, num_experts, expert_layers):
        super().__init__()
        self.num_experts = num_experts
        self.expert_layers = expert_layers

        self.experts = nn.ModuleList([self._create_expert(config) for _ in range(num_experts)])
        self.gate = nn.Linear(config.n_embd, num_experts)

    def _create_expert(self, config):
        layers = []
        for _ in range(self.expert_layers):
            layers.append(FlashAttention3(d_model=config.n_embd, n_heads=config.n_head, block_size_q=32, block_size_kv=32, num_blocks_kv=4))
            layers.append(nn.LayerNorm(config.n_embd))
            layers.append(MLP(config))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, T, C = x.size()
        
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        gate_probs = gate_probs.unsqueeze(-1)
        gate_probs = gate_probs.permute(0, 2, 1, 3)
        
        output = torch.sum(gate_probs * expert_outputs, dim=1)

        return output

# Define the BlockWithMoE module
class BlockWithMoE(nn.Module):
    def __init__(self, config, num_experts=4, expert_layers=2, block_size_q=32, block_size_kv=32, num_blocks_kv=4, device='cuda'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = FlashAttention3(d_model=config.n_embd, n_heads=config.n_head, block_size_q=block_size_q, block_size_kv=block_size_kv, num_blocks_kv=num_blocks_kv, device=device)
        self.dropout1 = nn.Dropout(config.dropout)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MixtureOfExperts(config, num_experts, expert_layers)
        self.dropout2 = nn.Dropout(config.dropout)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        attn_output = self.attn(x)
        x = x + attn_output
        x = self.dropout1(x)
        x = x + self.moe(self.ln_2(x))
        x = self.dropout2(x)
        x = x + self.mlp(self.ln_3(x))
        x = self.dropout3(x)
        return x
    
class GPTMoEMCTSPreTrainedModel(PreTrainedModel):
    config_class = GPTMoEMCTSConfig
    base_model_prefix = "gpt_moe_mcts"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

class GPTMoEMCTSModel(GPTMoEMCTSPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([BlockWithMoE(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(input_ids)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + (loss,) if loss is not None else (logits,)
            return output

        return {
            "logits": logits,
            "loss": loss,
        }
