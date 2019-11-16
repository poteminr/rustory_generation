# Language model for ru-story generation
## Corpus example 
Data from vk-group ["Pozor"](https://vk.com/styd.pozor) 

Total stories: 59173
>Если подумаете, что вы одинокий и вам станет грустно, то вспомните мужика, который в три часа ночи обнимал мусорный бак, при этом напевая песенку Агутина ты забудешь обо мне, на сиреневой луне. Жалко его стало.

## Used methods for text generation 
| Method | Code | Model | Choice Policy
|:----|:----|:----|:----|
| Transformer  | transformer.py | Transformer | Working on it
| Character based  | LM_char.ipynb | LSTM | Random choice of k most likely
| N-gram based | LM_ngram.ipynb | GRU | Max probability
| Word based | LM_word-based.ipynb | LSTM |  Random choice of k most likely



## Pre-trained embeddings
| Name | Comment |
|:----|:----|
| RuBert | Very hard to fine-tune | 

## Data preprocessing

- Emoji removed
- Ads removed
- Admin`s comments removed
- Tokenize or remove punctuation
- Stories lenght > 10 words

## Hardware
- 2 x GTX 1080
- Tesla P100-16GB


## Results 
I just started, the results will be later.