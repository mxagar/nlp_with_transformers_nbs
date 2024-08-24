"""This is a very simple implementation
of the Transfomer-Encoder, following the
book

    Natural Language Processing with Transformers
    
by Tunstall et al.

Dependencies:
- transformers
- torch

The configuration class from AutoConfig is the following:

    BertConfig {
    "_name_or_path": "bert-base-uncased",
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "gradient_checkpointing": false,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2"
    },
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_2": 2
    },
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.16.2",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
    }
"""
import torch
from math import sqrt
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


def scaled_dot_product_attention(query, key, value, mask=None):
    """Attention is implemented here.
    It is called self-attention because we use only the encoder hidden states
    to compute all similarities between tokens in the sequence simultaneously.
    That means, no decoder hidden states are used. The output is a set of
    contextualized embeddings, i.e., a weighted sum of all the embeddings,
    which captures the context information."""
    dim_k = query.size(-1)
    # Batch matrix-matrix product
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        # This masked option is for the Decoder part
        # not the Encoder. We can create a bottom-triangular matrix
        # with 1s (bottom part) and 0s (upper part) as follows:
        # mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    # This matrix multiplication produces a weighted sum of all value vectors
    # which is a set of contextualized embeddings
    return weights.bmm(value)


class AttentionHead(nn.Module):
    """Self-attention head."""
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs


class MultiHeadAttention(nn.Module):
    """It combines several attention heads (M = 12 in BERT base).
    Orange block in the original paper diagram."""
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    """Often called Positional Feed-Forward layer.
    Blue block in the original paper diagram."""
    def __init__(self, config):
        super().__init__()
        # The intermediate_sie is often 4x the hidden_size = embed_dim
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    

class TransformerEncoderLayer(nn.Module):
    """This class is a single encoder block which contains
    
    - a multi-head attention layer (with several attention heads + a linear layer),
    - normalization layers (inputs in the batch are transformed to have zero mean and unity variance),
    - a positional feed-forward layer,
    - and skip connections (to avoid gradient vanishing and enable deeper networks).
    
    This block is the grayed block in the original paper diagram,
    repeated Nx, being in BERT N = 12 (base) or 16 (large).
    
    Note that the arrangement in which we apply layer normalization and skip connections
    might vary:
    
    - Post-layer normalization (original paper): trickier to train.
    - Pre-layer normalization (most common now, applied here): more stable during training.
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class Embeddings(nn.Module):
    """This layer creates the embeddings from token input ids.
    It is applied only in the beginning.
    In addition, positional embeddings are added here, too; these are needed
    because the attention layer computes a weighted sum of all token embeddings,
    so the outputs loose the positional information.
    
    There are several ways to add positional information:
    
    - Learnable positional embeddings (applied here): a learnable embedding layer
    is added; this is the most popular approach nowadays if the dataset is large enough.
    - Absolute positional representations (original paper): static patterns
    consisting of modulates sine/cosine patterns; useful for small datasets.
    - Relative positional representations: the attention mechanism is modified incorporating
    the relative embedding positions.
    """
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, 
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# This is the Transformer-Encoder without any downstream/task head
# We need to pass to it a tensor with token ids, so the tokenization
# and the vocabulary need to be generated.
class TransformerEncoder(nn.Module):
    """The final but task-less Transformer-Encoder."""
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) 
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


# This is the Transformer-Encoder with a classification task head
class TransformerForSequenceClassification(nn.Module):
    """The final Transformer-Encoder with a classification task."""
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, x):
        x = self.encoder(x)[:, 0, :] # select hidden state of [CLS] token
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model_ckpt = "bert-base-uncased"
    text = "time flies like an arrow"

    # We need to tokenize our inputs to ids to pass them to the transformer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # The configuration class contains many parameters,
    # it's like a dictionary which contains all of them together
    # and the code is much cleaner
    config = AutoConfig.from_pretrained(model_ckpt)

    # Here, special tokens as CLS and SEP as excluded for simplicity
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    # inputs.input_ids: tensor([[ 2051, 10029,  2066,  2019,  8612]])
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    # Embedding(30522, 768)
    inputs_embeds = token_emb(inputs.input_ids)
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)
    
    multihead_attn = MultiHeadAttention(config)
    attn_outputs = multihead_attn(inputs_embeds)    
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)
    
    feed_forward = FeedForward(config)
    ff_outputs = feed_forward(attn_outputs)
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)
    
    encoder_layer = TransformerEncoderLayer(config)
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)
    
    embedding_layer = Embeddings(config)
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)
    
    # This is the Transformer-Encoder without any downstream/task head
    # We need to pass to it a tensor with token ids, so the tokenization
    # and the vocabulary need to be generated.
    encoder = TransformerEncoder(config)
    outputs = encoder(inputs.input_ids)
    # torch.Size([1, 5, 768]): (batch_size, seq_len, hidden_size = embed_dim)

    # This is the Transformer-Encoder with a classification task head
    config.num_labels = 3
    encoder_classifier = TransformerForSequenceClassification(config)
    classes = encoder_classifier(inputs.input_ids)
    # torch.Size([1, 3]): (batch_size, num_labels)