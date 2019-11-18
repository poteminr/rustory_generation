import youtokentome as yttm
import numpy as np

def split_data(data_batch, train_size=0.75):
    np.random.shuffle(data_batch)

    split_threshold = int(len(data_batch) * train_size)
    train_texts = data_batch[:split_threshold]
    test_texts = data_batch[split_threshold:]

    print('Total samples: {}'.format(len(train_texts) + len(test_texts)))
    print('')
    print('Traning size: {} | Validating size: {}'.format(len(train_texts), len(test_texts)))

    return train_texts, test_texts


def _save_text(texts, out_file):
    with open(out_file, 'w') as outf:
        outf.write('\n'.join(texts))

def get_bpe_tokenizer(train_texts, train_txt_path, bpe_model_name, vocab_size):
    _save_text(train_texts, train_txt_path)
    yttm.BPE.train(data=train_txt_path, vocab_size=vocab_size, 
                   model=bpe_model_name)
    
    tokenizer = yttm.BPE(bpe_model_name)

    return tokenizer

def get_unknown_ngrams(test_token_ids):
    unknown_subwords_in_test = sum(1 for text in test_token_ids for token_id in text if token_id == 1)
    print('Unknown n-grams in validation set: ', unknown_subwords_in_test)
