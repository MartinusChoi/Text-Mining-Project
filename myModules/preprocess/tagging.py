from tqdm.notebook import tqdm

from konlpy.tag import Kkma
import nltk

def tagging(articles, mode):
    tagged = []

    if mode == 'ST':
        for article in tqdm(articles):
            tagged.append(nltk.pos_tag(article))
    elif mode == 'TT':
        kkma = Kkma()
        for article in tqdm(articles):
            tagged.append(kkma.pos(article))
    
    return tagged