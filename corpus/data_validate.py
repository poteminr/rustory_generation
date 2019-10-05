import re, string, unicodedata
from tqdm import tqdm_notebook
import numpy as np

ban_list = ["vk.com", "Бот", "bot", "club"]
def check_valid(data_dct: dict):
    
    if "attachments" in data_dct.keys():
        return False
    
    text = data_dct['text']
    
    for cond in ban_list:
            if len(re.findall(cond, text)) > 0:
                return False

    return True

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_punctuation(tokens):
    new_words = []
    for word in tokens:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_emoji(string):
    emoji_pattern = re.compile("["
u"\U0001F600-\U0001F64F"  
u"\U0001F300-\U0001F5FF"  
u"\U0001F680-\U0001F6FF"  
u"\U0001F1E0-\U0001F1FF"  
u"\U00002702-\U000027B0"
u"\U000024C2-\U0001F251"
"]+", flags=re.UNICODE)
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    string = emoji_pattern.sub(r'', string)
    string = RE_EMOJI.sub(r'', string)
    return string


def data_preproc(data: list, remove_punct=True, remove_smiles=True, tokenize_punct=False, is_lower=True, len_threshold=10):
    assert not (remove_punct == True and tokenize_punct == True), "Use one of this parameters"

    full_data = []
    for sent in tqdm_notebook(data):
        if check_valid(sent):
            text = sent['text']

            text = text.replace("\n", " ").replace("  ", " ")
            text = text.replace("_", "")

            if is_lower:
                text = text.lower()

            if remove_smiles:
                text = remove_emoji(text)

            if tokenize_punct:
                text_tokens = re.findall(r"[\w']+|[.,!?;]", text)
            else:
                text_tokens = text.split(" ")

            if remove_punct:
                text_tokens = remove_punctuation(text_tokens)

            if "admin" in text_tokens:
                text_tokens = text_tokens[:text_tokens.index("admin")]

            if len(text_tokens) > len_threshold:
                text_tokens = np.array(text_tokens)
                full_data.append(text_tokens)
        
    print("Samples: {} | Removed samples: {}".format(len(full_data), len(data) - len(full_data)))
    return np.array(full_data)