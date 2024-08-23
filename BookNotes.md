# Natural Language Processing with Transformers: My Notes

These are my notes of the book [Natural Language Processing with Transformers, by Lewis Tunstall, Leandro von Werra and Thomas Wolf (O'Reilly)](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).

Table of contents:

- [Natural Language Processing with Transformers: My Notes](#natural-language-processing-with-transformers-my-notes)
  - [Setup](#setup)
    - [Colab Setup](#colab-setup)
  - [Chapter 1: Hello Transformers](#chapter-1-hello-transformers)
    - [Key points](#key-points)
    - [Notebook](#notebook)
    - [List of papers](#list-of-papers)
  - [Chapter 2: Text Classification](#chapter-2-text-classification)
    - [Key points](#key-points-1)
    - [Notebook](#notebook-1)
    - [List of papers](#list-of-papers-1)
  - [Chapter 3: Transformer Anatomy](#chapter-3-transformer-anatomy)
    - [Key points](#key-points-2)
      - [The Encoder](#the-encoder)
      - [The Decoder](#the-decoder)
      - [Pytorch Implementation](#pytorch-implementation)
      - [Transformers](#transformers)
    - [Notebook](#notebook-2)
    - [List of papers](#list-of-papers-2)
  - [Chapter 4: Multilingual Named Entity Recognition](#chapter-4-multilingual-named-entity-recognition)
  - [Chapter 5: Text Generation](#chapter-5-text-generation)
  - [Chapter 6: Summarization](#chapter-6-summarization)
  - [Chapter 7: Question Answering](#chapter-7-question-answering)
  - [Chapter 8: Making Transformers Efficient in Production](#chapter-8-making-transformers-efficient-in-production)
  - [Chapter 9: Dealing with Few to No Labels](#chapter-9-dealing-with-few-to-no-labels)
  - [Chapter 10: Training Transformers from Scratch](#chapter-10-training-transformers-from-scratch)
  - [Chapter 11: Future Directions](#chapter-11-future-directions)

See also:

- [mxagar/tool_guides/hugging_face](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [mxagar/generative_ai_udacity](https://github.com/mxagar/generative_ai_udacity)
- [mxagar/generative_ai_book](https://github.com/mxagar/generative_ai_book)
- [mxagar/nlp_guide](https://github.com/mxagar/nlp_guide)

## Setup

I used Google Colab, so no setup was needed apart from the local installations in each notebook.

### Colab Setup

The following lines need to be added and modified in each notebook.

```python
# Uncomment and run this cell if you're on Colab or Kaggle
!git clone https://github.com/mxagar/nlp_with_transformers_nbs.git
%cd nlp_with_transformers_nbs
from install import *
install_requirements(is_chapter2=True)

# Log in to HF with HF_TOKEN
from huggingface_hub import notebook_login
notebook_login()
```

## Chapter 1: Hello Transformers

### Key points

- Two major papers that led to the rise of the Transformers in NLP
  - Attention is all you need (Vaswani, 2017): Transformer model
    - Encoder-decoder architecture.
    - Self-attention to replace LSTMs, so that sequential tasks are parallelizable.
  - ULMFiT (Howard, 2017): a language model trained on a large corpus can be re-adapted for smaller corpora and other downstream tasks.
    - Transfer learning was validated also for NLP; until then, only worked in CV.
- After those papers, the two most important models were published:
  - GPT: decoder, generative model.
  - BERT: encoder.
- Encoder-decoder framework
  - Before the transformers, LSTMs were SOTA.
  - LSTMs have a hidden state which accumulates previous inputs.
  - Encoder-decoder architectures 
    - Enconder would receive an sequence of words, pass it one-by-one through the LSTM layers and obtain a final hidden state.
    - The final hidden state which would be the seed for the decoder, which would consists again of LSTM layers that would produce an output sequence.
    - That way, sequence-to-sequence tasks can be carried out, e.g., language translation or summarization.
        ![Encoder-Decoder](./images/chapter01_enc-dec.png)
    - Problems:
      - (1) Hidden state after a long sequence forgets beginning.
      - (2) We need to pass the words of the sequence one by one.
    - Solutions:
      - (1) Attention was developed (Bahdanau, 2014) and applied: attention layers learn to apply relevance weights of values/vectors; thus, instead of taking the last hidden state, it is possible to take all intermediate hidden states and apply later attention to them.
      - (2) Transformer architecture (Viswani, 2017) does not have LSTMs but MLPs which support processing the complete sequence all at once. It additionally has *self-attention*.
        ![Self-Attention](./images/chapter01_self-attention.png)
- Transfer Learning
  - Common in Computer Vision, but it was shown to work by ULMFiT (Howard, 2017).
    - In CV, fine-tuned models (body/backbone + head) work better than models trained from-scratch.
    - A work related to ULMFiT: ELMo.
  - The ULMFiT framework (they used LSTMs and the *predict-next-word* task, aka. **language modeling**):
    - Pretraining: language modeling (i.e., predict next word) with Wikipedia (large corpus).
    - Domain adaptation: language modeling (i.e., predict next word) with IMDB dataset (small corpus).
    - Fine-tuning: model is fine-tuned for a new downstream task using the adaptation dataset (IMDB), sentiment classification of moview reviews.
- First new models after Transformers were discovered:
  - GPT: decoder, generative. Trained on BookCorpus.
  - BERT: encoder. Trained on BookCorpus. Masked language modeling, predicting the masked word in a test, e.g. *I looked at my [MASK] and saw that it was late*.
- HuggingFace Transformers; examples with `pipeline`: a message to Amazon requesting the correct order is shown, and several models applied:
  - Text classification (sentiment, multi-class, multi-label)
  - Named Entity Recognition (NER): persons, locations, organizations, brands, etc.
  - Question answering: extractive QA, answers provided and location in text used for formulating answer.
  - Summarization.
  - Translation (to German).
  - Text generation: answer from Amazon is generated.
- Tools from Hugging Face:
  - Hub: Models, Datasets, Metrics
  - Libraries: Tokenizers (Rust backend), Transformers (compatible with Pytorch and Tensorflow), Datasets (compatible with Pandas and Numpy), Accelerate (abstraction of training loops for faster deployment).

### Notebook

[`01_introduction.ipynb`](./01_introduction.ipynb):

- Pipelines
- Text classification
- Named Entity Recognition
- Question Answering
- Summarization
- Translation
- Text Generation

### List of papers

- Encoder-decoder (Cho, 2014): [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- Attention (Bahdanau, 2014): [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Transformer (Vaswani, 2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- ULMFiT (Howard, 2018): [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- ELMo (Peters, 2018): [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
- GPT (Radford, 2018): [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- BERT (Devlin, 2018): [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Chapter 2: Text Classification

In this chapter the DistilBERT model is adapted and fine-tuned to perform tweet text classification (6 emotions detected); all the NLP pipeline is thoroughly explained. Recall that DistilBERT is originally trained to predict a masked word in a sequence.

### Key points

The complete chapter is about the implementation of the notebook [`02_classification.ipynb`](./02_classification.ipynb). Unfortunately, the notebook does not work right away due to versioning issues. However, I wrote down the most important insights.

A distilled BERT model is taken (encoding transformer) and a classifier is used with it. This classifier is used to predict 6 emotions in a dataset consisting of tweets. The model is trained in two ways:

- The embedding vectors are used to train a logistic regression model; poor performance.
- A classifier head is attached to the transformer and all the weights are trained; i.e., we perform fine-tuning. The performance is much better.

Finally, we can push the weights of the trained model to Hugging Face.

Key points:

- DistilBERT: a distilled version of BERT, much smaller (faster), but similar performance.
- Dataset: `"dair-ai/emotion"`. Six emotions assigned to tweets: `anger, love, fear, joy, sadness, surprise`.
  - **Note**: I could not download the dataset with the provided versions; I did not spend much time fixing the issue...
- We can `list_datasets`.
- We can `load_dataset` given its identifier string.
  - Then, the dataset is a dictionary which contains the keys `"train"` and `"test"`.
  - Each row is a dictionary, because the values are columnar, using Apache Arrow.
  - It is also possible to load CSV or other/similar files using `load_dataset`.
  - Dataset objects can be converted to Pandas with `.set_format(type="pandas")`.
- EDA with the dataset
  - The dataset seems highly imbalanced, but we leave it so.
  - Tweet length vs emotion is plotted in box plots.
  - Later, hex-plots are performed with text embeddings for each emotion. The embeddings are mapped to 2D using UMAP.
- Tokenization and numericalization (`token2id` dictionary)
  - Character tokenization: split by character
    - Simplest scheme, smallest vocabulary size.
    - Problem: words need to be learned from characters.
  - Word tokenization: split by white spaces.
    - Usually, stemming and lemmatizaton are also applied to reduce number of tokens.
    - Much larger vocabulary size.
    - Unknown tokens/words are mapped to `UNK` token.
    - Large vocabularies require many weights in the NN
  - **Subword tokenization**: words are split, combining the best features of the previous two.
    - We can deal with misspellings
    - It is learned from a pretraining corpus.
    - Many subword tokenizers; **WordPiece** used by BERT and DistilBERT.
  - `AutoTokenizer` provides the correct tokenizer for each model; we can also manually instantiate the proper tokenizer for our model (e.g., `DistilBertTokenizer`).
  - Common methods of the tokenizer:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_text = tokenizer(text)
    tokens = tokenizer.convert_is_to_tokens(encoded_text.input_ids)
    tokenizer.vocab_size # 30k
    tokenizer.model_max_length # 512, context size
    ```
  - Common features of the used subword tokenizer:
    - Words are split in endings and `##` is appended to the suffixes: `tokenizing -> token, ##izing`.
    - Special tokens are defined: `[PAD], [UNK] (unknown), [CLS] (start), [SEP] (end), [MASK]`.
    - We tokenize the entire dataset by padding if sequence is shorter and truncating otherwise.
    - The output ids have a value `attention_mask`, which is `1` usually, `0` if the token is padded.
    - To tokenize the entire dataset, we define a `tokenize()` function and `map()` it to the dataset.
- Approach 1: Transformers as feature extractors
  - Token embeddings are generated and saved as a dataframe.
    - The selected embeddings are the hidden states from the transformer; for classification, the hidden states of the first token `[CLS]`.
    - We can see that the dimension of the output vector is `(batch_size, seq_len, hidden_size)`; the `hidden_size = 768`.
  - Then, a separate model is trained with the dataframes: logistic regression (choice in the chapter), random forest, XGBoost, etc.
  - The baseline of the classification is the random `DummyClassifier` from Scikit-Learn. Always do this! It is helpful in imbalanced multi-classes cases.
  - The result is a score of 63% (F1); the confusion matrix shows the miss-classifications.
- Approach 2: Add a classification head and fine-tune everything.
  - A classification head which is differentiable is added; that can be done automatically with `AutoModelForSequenceClassification`, for which we specify `num_labels = 6`.
  - We train everything with a `Trainer` instance, which takes:
    - `TrainingArguments`: a class with hyperparameters
    - A custom defined `compute_metrics` function; it receives a `EvalPrediction` object and returns a dictionary with the metric values.
    - The train and eval datasets
    - The `tokenizer`
  - After training, we can `trainer.predict()`: 91% (F1), much better!
  - Error analysis: if we pass a validation split to the `predict` method, we get the loss of each observation, thus, we can list the mis-predictions by loss. Then, we perform an error analysis to detect:
    - Wrong labels.
    - Quirks of the dataset: spacial characters can lead to wrong classifications, maybe we discover new ways of cleaning the data, etc.
    - Similarly, we should look at the samples with the lowest losses.
- Interoperability between frameworks: in some cases the weights of some models are in Pytorch and we'd like to have them in Tensorflow; we can do that using Hugging Face.
- It is also possible to train with Keras.
- We can push the weights of the trained model using `trainer.push_to_hub()`.
- We need to log in to Hugging Face.

### Notebook

[`02_classification.ipynb`](./02_classification.ipynb)

Unfortunately, the notebook does not work right away due to versioning issues.
However, I wrote down the most important insights in the section before.

### List of papers

- DistilBERT (Sanh, 2019): [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

## Chapter 3: Transformer Anatomy

This chapter explains the architecture of the transformer.

### Key points

The original Transformer architecture (Vaswani et al., 2017) is based on the **encoder-decoder** architecture:

- The encoder converts input sequence (tokens) into embedding vectors, i.e., **hidden states** or **context**.
- The decoder converts those hidden states into an output sequence.

![Original Transformer Architecture](./assets/transformer_architecture.png)
![Eencoder-Decoder](./images/chapter03_transformer-encoder-decoder.png)

The original model was used for translation tasks, i.e., from an input sequence in a language to the output sequence in another language. However, the original architecture was quickly split into 3 families:

- Encoder-only: convert input sequence into an embedding that can be used in downstream tasks (classification, etc.).
  - Examples: BERT, DistilBERT, RoBERTa
  - Training is done with masked tokens, thus, left and right parts of a token are used to compute the representation, i.e., **bi-directional attention**.
- Decoder-only: autocomplete a sequence by predicting the most probable next word/token.
  - Example: GPT
  - Training is done to predict next word, so only the left pert of a token is used, aka. **causal or autoregressive attention**.
- Encoder-decoder: from sequence to sequence.
  - Examples: BART, T5.

Some other features of the architecture:

- The line between the 3 branches is a bit blurry: decoders can also translate and encoders can also summarize!
- The components of the architecture are quite simple; we have blocks that contain:
  - Attention layers: similarities of the tokens in the sequence are computed simultaneously (i.e., dot product) and used to weight and sum the embeddings in successive steps.
  - Feed-forward layers: linear transformations of embeddings.
  - Positional embeddings/encodings: since the transformed embeddings are continuously the sum of weighted embeddings, we need to add the lost position information somehow; that can be achieved in different ways using positional embeddings.
  - Normalization
- Both the encoder and the decoder use similar blocks; however
  - The encoder transforms embeddings in successive steps to produce the output embeddings.
  - The decoder takes the output embeddings from the encoder as well as some hidden states and generates a new sequence, which is finished when the `EOS` token (end-of-sentence) emerges.

#### The Encoder

Components (in order):

- **Embedding layer**: tokens are converted into vectors; BERT, `embed_dim = 768 (base) or 1024 (large)`.
- **Positional embeddings/encodings** are added: since the attention layer computes a weighted sum of all tokens in the sequence, we need to encode the position of the tokens.
- N encoder blocks stacked serially; BERT, N = 12 (base) or 24 (large).  
  One **encoder block** has:
    - **Multi-head self attention + Concatenation + Feed-forward**.
    - Each multi-head attention layer has M self-attention heads; BERT, M = 12 (base) or 16 (large).
      - Each **(self-)attention head** decomposes the input embedding sequences into M sequences that are later concatenated again to form an updated embedding. After decomposition, hidden embedding sequences are created, upon which self-attention is applied:
        - We transform the original embedding into Q (query), K (key), V (value). The transformation is performed by a linear/dense layer, which is learned.
        - Q and K are used to compute a similarity score between token embedding against token embedding (dot product).
        - The similarity is used as a weight to sum all token embeddings from V to yield a new set of hidden embeddings. These are called **contextualized embeddings**, because they contain context information, i.e., the information of the surrounding embeddings. These updates and context integration solves issues like homonyms or difference in word order between different languages.

The attention block is called *self-attention block* because all attention/similarity weights are computed simultaneously for the entire sequence using only the embeddings themselves.

The output embeddings from each encoder block have the same size as the input embeddings, so the encoder block stack has the function of updating those embeddings. This is the summary of the sizes:

- Input embedding sequence: `(batch_size, seq_len, embed_dim)`.
- Embeddings being transformed inside a self-attention layer: `(batch_size, seq_len, head_dim = embed_dim/M)`.
- Embeddings being transferred from one multi-head block to another: `(batch_size, seq_len, hidden_dim = embed_dim)`.

![Transformer Architecture Components](./assets/Transformer_Architecture_Components.png)

In addition to the already mentioned components, each encoder block has also:

- Layer normalization: inputs/outputs are normalized to mean 0, std 1. This normalization can be pre-layer (original) or post-layer (most common nowadays).
- Skip connections: previous embeddings (before transformations/updates) are added. These skip/residual connections
  - Minimize the gradient vanishing problem.
  - Enable deeper networks.
  - Improve the optimization of the loss function.
- Positional feed-forward layer: two linear/dense matrices (followed by a dropout) further transform the embeddings. Usually, 
  - the first transformation increases 4x the size of the embeddings 
  - and the second scales back the embeddings to their original size.

![Layer normalization](./images/chapter03_layer-norm.png)

#### The Decoder


#### Pytorch Implementation

#### Transformers


### Notebook

[`03_transformer-anatomy.ipynb`](./03_transformer-anatomy.ipynb)

### List of papers

- Transformer (Vaswani et al., 2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- ELMo Contextualized embeddings (Peters et al.): [Deep Contextualized Word Representations]()

## Chapter 4: Multilingual Named Entity Recognition

## Chapter 5: Text Generation

## Chapter 6: Summarization

## Chapter 7: Question Answering

## Chapter 8: Making Transformers Efficient in Production

## Chapter 9: Dealing with Few to No Labels

## Chapter 10: Training Transformers from Scratch

## Chapter 11: Future Directions

