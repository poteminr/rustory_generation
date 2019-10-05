# Language model for ru-story generation
## Corpus example 
Data from vk-group ["Pozor"](https://vk.com/styd.pozor) 

Total samples: 59173
>Если подумаете, что вы одинокий и вам станет грустно, то вспомните мужика, который в три часа ночи обнимал мусорный бак, при этом напевая песенку Агутина ты забудешь обо мне, на сиреневой луне. Жалко его стало

## Used methods for text generation 
| Method | Code | Model |
|:----|:----|:----|
| N-gram based | LM_ngram.ipynb | GRU | 
| Word based | LM_word-based.ipynb | GRU |
| Character based  | None | None |


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
