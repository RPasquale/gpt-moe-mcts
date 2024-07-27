from transformers import GPT2Tokenizer
from modeling_gpt_moe_mcts import GPTMoEMCTSModel
from configuration_gpt_moe_mcts import GPTMoEMCTSConfig

# Initialize configuration
config = GPTMoEMCTSConfig()

# Initialize model
model = GPTMoEMCTSModel(config)

# Initialize tokenizer (using GPT2Tokenizer as a base)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Prepare input
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
outputs = model(**inputs)

# Get the predicted next token
next_token_logits = outputs.logits[0, -1, :]
next_token = next_token_logits.argmax()

# Decode the predicted token
predicted_text = tokenizer.decode(next_token)

print(f"Input: {text}")
print(f"Predicted next token: {predicted_text}")