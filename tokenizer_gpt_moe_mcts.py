from transformers import GPT2Tokenizer

class GPTMoEMCTSTokenizer(GPT2Tokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            merges_file,
            **kwargs
        )
