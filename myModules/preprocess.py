############## Moduel Import ##############

import re
from tqdm.notebook import tqdm

from konlpy.tag import Kkma
import nltk

from tensorflow.keras.preprocessing.text import text_to_word_sequence

############## Cleaning ##############

def lowerCase(text):
    return text.lower()

def cleanST(text):
    text = lowerCase(text)
    return re.sub("[^a-z'\. ]", " ", text)

def cleanTT(text):
    return re.sub('[^A-Za-z가-힣 ]', '', text)

def cleaning(texts, mode):
    result=[]

    if mode == 'ST':
        for text in texts:
            result.append(cleanST(str(text)))
    elif mode == 'TT':
        for text in texts:
            result.append(cleanTT(str(text)))
    
    return result

######################################

############## StopWords ##############

def removeStopWords_ST(articles, stopwords=[], new_stopwords=[]):
    removed = []
    
    if len(new_stopwords) != 0:
        stopwords.extend(new_stopwords)

    for article in articles:
        removed.append([token for token in article if token not in stopwords])

    return removed

def removeStopWord_TT(articles, stop_tag_list, Kor_stopwords):
    removed = []

    for article in tqdm(articles):
        arr = []
        for tag in article:
            if (tag[1] not in stop_tag_list) & (tag[0] not in Kor_stopwords) & (len(tag[0]) != 1):
                arr.append(tag)
        removed.append(arr)

    return removed

######################################

############## tagging ##############

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

######################################

############## tokenizing ##############

def tokenizing_ST(articles, tokenizer):
    tokenized = []

    for article in articles:
        tokenized.append(tokenizer.tokenize(article))
    
    return tokenized

def tokenizing_TT(articles, tagList, pos='all'):
    if pos == 'noun':
        nouns = []
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[0]:
                    tags.append(str(tag[0]))
            nouns.append(tags)
        return nouns
    
    elif pos == 'verb':
        verb=[]
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[2]:
                    tags.append(str(tag[0]))
            verb.append(tags)
        return verb
    
    elif pos == 'adjective':
        adjective = []
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[3]:
                    tags.append(str(tag[0]))
            adjective.append(tags)
        return adjective
    
    elif pos == 'all':
        all = []
        for article in articles:
            tokens = []
            for tag in article:
                tokens.append(str(tag[0]))
            all.append(tokens)
        return all
    
    else :
        print("Invalid POS mode! must be one of : 'none', 'verb', 'adjective'")
        return -1

def keras_tokenizer(articles):
    result = []

    for article in articles:
        result.append(text_to_word_sequence(article))
    
    return result
    
######################################

def extract_some_pos_ST(articles, tagList, pos_list=['noun', 'pronoun', 'verb', 'adjective']):
    result = []

    for article in articles:
        tags = []
        for tag in article:
            if 'noun' in pos_list:
                if tag[1] in tagList.Eng_tag[0]:
                    tags.append(str(tag[0]))
            if 'verb' in pos_list:
                if tag[1] in tagList.Eng_tag[2]:
                    tags.append(str(tag[0]))
            if 'adjective' in pos_list:
                if tag[1] in tagList.Eng_tag[3]:
                    tags.append(str(tag[0]))
        result.append(tags)
    
    return result
    
