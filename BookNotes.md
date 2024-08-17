# Natural Language Processing with Transformers: My Notes

These are my notes of the book [Natural Language Processing with Transformers, by Lewis Tunstall, Leandro von Werra and Thomas Wolf (O'Reilly)](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/).

Table of contents:

- [Natural Language Processing with Transformers: My Notes](#natural-language-processing-with-transformers-my-notes)
  - [Setup](#setup)
  - [Chapter 1: Hello Transformers](#chapter-1-hello-transformers)
    - [Key points](#key-points)
    - [Notebook](#notebook)
    - [List of papers](#list-of-papers)
  - [Chapter 2: Text Classification](#chapter-2-text-classification)
    - [Key points](#key-points-1)
    - [Notebook](#notebook-1)
    - [List of papers](#list-of-papers-1)
  - [Chapter 3: Transformer Anatomy](#chapter-3-transformer-anatomy)
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

### Notebook

### List of papers

- DistilBERT (Sanh, 2019): [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)


## Chapter 3: Transformer Anatomy

## Chapter 4: Multilingual Named Entity Recognition

## Chapter 5: Text Generation

## Chapter 6: Summarization

## Chapter 7: Question Answering

## Chapter 8: Making Transformers Efficient in Production

## Chapter 9: Dealing with Few to No Labels

## Chapter 10: Training Transformers from Scratch

## Chapter 11: Future Directions

