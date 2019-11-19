# Language model for ru-story generation
## Corpus example 
Data from vk-group ["Pozor"](https://vk.com/styd.pozor) 

Total stories: 59173
>Если подумаете, что вы одинокий и вам станет грустно, то вспомните мужика, который в три часа ночи обнимал мусорный бак, при этом напевая песенку Агутина ты забудешь обо мне, на сиреневой луне. Жалко его стало.

## Used methods for text generation 
| Method | Code | Model | Choice Policy
|:----|:----|:----|:----|
| Transformer  | Transformer based/Transformer.ipynb | Transformer | Max probability
| Character based  | LMs/LM_char.ipynb | LSTM | Random choice of k most likely
| N-gram based | LMs/LM_ngram.ipynb | GRU | Max probability
| Word based | LMs/LM_word-based.ipynb | LSTM |  Random choice of k most likely



## Pre-trained embeddings and tokenizers
| Name | Comment | 
|:----|:----|
| RuBert | Very hard to fine-tune, not used.|
| YouTokenToMe |I think it's better than char-level tokenizer|

## Data preprocessing

- Emoji removed
- Ads removed
- Admin`s comments removed
- Tokenize or remove punctuation
- Stories > 10 words
- Stories < 200 chars (for Transformer)
- BPE tokenizer (for Transformer)

## Hardware
- GTX 1080x2
- Tesla P100-16GB


## Results 
I just started, the results will be later.