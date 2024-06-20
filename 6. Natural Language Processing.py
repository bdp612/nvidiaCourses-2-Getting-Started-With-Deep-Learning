import torch

# Load the tokenizer and model for Masked Language Modeling from Huggingface's transformers library
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

def get_segment_ids(indexed_tokens):
    """
    Generate segment IDs for the input tokens.
    Segment IDs are used to differentiate between sentences in BERT input.
    """
    segment_ids = []
    segment_id = 0
    for token in indexed_tokens:
        if token == sep_token:  # Increment segment ID at each [SEP] token
            segment_id += 1
        segment_ids.append(segment_id)
    segment_ids[-1] -= 1  # The last [SEP] token is ignored
    return torch.tensor([segment_ids]), torch.tensor([indexed_tokens])

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
tokenizer.convert_ids_to_tokens([str(token) for token in indexed_tokens])
tokenizer.decode(indexed_tokens)
cls_token = 101  # CLS token ID for BERT
sep_token = 102  # SEP token ID for BERT

# Mask a token (here, the token at position 5) to predict it
masked_index = 5
segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)
tokens_tensor = torch.tensor([indexed_tokens])
indexed_tokens[masked_index] = tokenizer.mask_token_id  # Replace the token at masked_index with [MASK] token
tokenizer.decode(indexed_tokens)

# Predict the masked token
with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

# Sentence and predicted word
print("Sentence with missing word: '" + tokenizer.decode(indexed_tokens) + "'")
print("Missing word: '" + predicted_token + "'")

# Question Answering setup
text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)

# Load the model for Question Answering
question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# Extract the answer from the predicted start and end positions
answer_sequence = indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1]
question_answering_tokenizer.convert_ids_to_tokens(answer_sequence)

# Answering the question from text_2
print("Prompt sentence: '" + text_1 + "'")
print("Question: '" + text_2 + "'")
print("Answer: '" + question_answering_tokenizer.decode(answer_sequence) + "'")