from transformers import PretrainedConfig

class GPTMoEMCTSConfig(PretrainedConfig):
    model_type = "gpt_moe_mcts"

    def __init__(
        self,
        vocab_size=50257,
        block_size=512,
        n_layer=6,
        n_head=4,
        n_embd=256,
        dropout=0.2,
        num_experts=3,
        expert_layers=3,
        block_size_q=32,
        block_size_kv=32,
        num_blocks_kv=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.num_experts = num_experts
        self.expert_layers = expert_layers
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.num_blocks_kv = num_blocks_kv