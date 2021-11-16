############## Moduel Import ##############

import re
from tqdm.notebook import tqdm

from konlpy.tag import Kkma

############## Cleaning ##############
def clean(text):
    return re.sub('[^A-Za-z가-힣 ]', '', text)

def cleaning(data):
    result=[]

    for text in data:
        result.append(clean(str(text)))
    
    return result

######################################

############## StopWords #############

def remove_stopword(data, stop_tag_list, Kor_stopwords):
    result = []

    for tags in tqdm(data):
        arr = []
        for tag in tags:
            if (tag[1] not in stop_tag_list) & (tag[0] not in Kor_stopwords) & (len(tag[0]) != 1):
                arr.append(tag)
        result.append(arr)

    return result

######################################

############## tagging ##############

def tagging(articles):
    result = []

    kkma = Kkma()

    for article in tqdm(articles):
        result.append(kkma.pos(article))
    
    return result

######################################

############## tokenizing ##############
def tokenizing(data, tagList, pos='all'):

    if pos == 'noun':
        nouns = []
        for article in data:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[0]:
                    tags.append(str(tag[0]))
            nouns.append(tags)
        return nouns
    
    elif pos == 'verb':
        verb=[]
        for article in data:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[2]:
                    tags.append(str(tag[0]))
            verb.append(tags)
        return verb
    
    elif pos == 'adjective':
        adjective = []
        for article in data:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[3]:
                    tags.append(str(tag[0]))
            adjective.append(tags)
        return adjective
    
    elif pos == 'adverb':
        adverb = []
        for article in data:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[4]:
                    tags.append(str(tag[0]))
            adverb.append(tags)
        return adverb            
    
    elif pos == 'all':
        all = []
        for article in data:
            tokens = []
            for tag in article:
                tokens.append(str(tag[0]))
            all.append(tokens)
        return all
    
    else :
        print("Invalid POS mode! must be one of : 'noun', 'verb', 'adjective', 'adverb', 'all'")
        return -1
    
######################################