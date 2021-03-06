############## Module Import ##############

import re
from tqdm.notebook import tqdm
import pickle
import pandas as pd

import nltk
import spacy

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from myModules.utils import merge

############## Cleaning ##############

def lowerCase(text):
    return text.lower()

def clean(text):
    text = lowerCase(text)
    return re.sub("[^a-z'\. ]", " ", text)

def cleaning(data):
    result=[]

    for text in data:
        result.append(clean(str(text)))
    
    return result

######################################

############## StopWords ##############

def remove_stopwords(data, stopwords=[], new_stopwords=[]):
    result = []
    
    if len(new_stopwords) != 0:
        stopwords.extend(new_stopwords)

    for article in data:
        result.append([token for token in article if token not in stopwords])

    return result

######################################

############## tagging ##############

def tagging(data):
    result = []

    for article in tqdm(data):
        result.append(nltk.pos_tag(article))    
    
    return result

######################################

############## tokenizing ##############

def tokenizing(data, tokenizer):
    result = []

    for article in data:
        result.append(tokenizer.tokenize(article))
    
    return result

def keras_tokenizer(data):
    result = []

    for article in data:
        result.append(text_to_word_sequence(article))
    
    return result
    
######################################

############## select some pos ##############

def select_some_pos(data, tagList, pos_list=['noun', 'adverb', 'verb', 'adjective']):
    result = []

    for article in data:
        tags = []
        for tag in article:
            if ('noun' in pos_list) & (tag[1] in tagList.Eng_tag[0]): tags.append(str(tag[0]))

            if ('verb' in pos_list) & (tag[1] in tagList.Eng_tag[2]): tags.append(str(tag[0]))

            if ('adjective' in pos_list) & (tag[1] in tagList.Eng_tag[3]): tags.append(str(tag[0]))

            if ('adverb' in pos_list) & (tag[1] in tagList.Eng_tag[4]): tags.append(str(tag[0]))
        result.append(tags)
    
    return result

######################################

############## dealing with dot and apostrophe ##############

class dot_and_apostrophe:
    def __init__(self, data):
        self.data = data
    
    def token_with_apostrophe(self):
        # apostrophe??? ?????? token ??? ?????? ??????
        apostrophe = []

        for tokens in self.data:
            for token in tokens:
                if "'" in token : apostrophe.append(token)
        
        self.apostrophes = set(apostrophe)

        print(f"apostrophe??? ?????? token : \n{self.apostrophes}")
    
    def token_with_dot(self):
        # dot??? ?????? token ??? ?????? ??????
        dot = []

        for tokens in self.data:
            for token in tokens:
                if "." in token : dot.append(token)
        
        self.dots = set(dot)

        print(f"dot??? ?????? token : \n{self.dots}")
        
    def set_exception(self, apostrophe_exception, dot_exception):
        # dot??? apostrophe??? ???????????? ?????? ?????? ??????
        self.apostrophe_exception = apostrophe_exception
        self.dot_exception = dot_exception
    
    def print_exception(self):
        # ????????? ?????? ?????? ??????
        print(f"apostrophe exceptions : \n{self.apostrophe_exception}")
        print(f"dot exceptions : \n{self.dot_exception}")
    
    def remove_apostrophe(self, data):
        # ?????? ?????? ?????? apostorphe??? ?????? token?????? symbol??? ??????
        result = []
        processed = []

        for tokens in data:
            arr = []
            for token in tokens:
                if token not in self.apostrophe_exception:
                    if not token.isalnum() : 
                        if "." not in token : processed.append(token)
                    # dot??? ???????????? ??????. -> ????????????????????? ???????????????
                    arr.append(re.sub("[^a-z\.]", "", token))
                else : arr.append(token)
            result.append(arr)
        
        processed = set(processed)

        print(f"Processed Tokens : \n{processed}")
        
        return result
    
    def remove_dot(self, data):
        # ?????? ?????? ?????? dot??? ?????? token?????? symbol??? ??????
        result = []
        processed = []

        for tokens in data:
            arr = []
            for token in tokens:
                if token not in self.dot_exception:
                    if not token.isalnum() : 
                        if "'" not in token : processed.append(token)
                    # apostrophe??? ???????????? ??????. -> ????????????????????? ??????
                    arr.append(re.sub("[^a-z']", "", token))
                else : arr.append(token)
            result.append(arr)
        
        processed = set(processed)

        print(f"Processed Tokens : \n{processed}")
        
        return result
    
    def check_invalid_tokens(self, data):
        # ??????????????? Token??? ?????? ??????????????? ?????? Token?????? ????????? ??????
        invalid_tokens = []

        for tokens in data:
            for token in tokens:
                if not token.isalnum() : invalid_tokens.append(token)
                elif len(token) == 1 : invalid_tokens.append(token)
        
        invalid_tokens = set(invalid_tokens)
        exception = set(self.apostrophe_exception).union(set(self.dot_exception))
        self.invalid_symbol = invalid_tokens.difference(exception)

        if len(self.invalid_symbol) == 0:
            print("There is no invalid symbol")
        else :
            print(f"Remaining invalid Symbol : {self.invalid_symbol}")
    
    def remove_invalid_tokens(self, data):
        # ???????????? ???????????? + ????????? 1??? token?????? ??????
        
        result = []
        removed = []

        for tokens in data:
            arr = []
            for token in tokens:
                if len(token) == 1 : removed.append(token)
                elif token in self.invalid_symbol : removed.append(token)
                else : arr.append(token)
            result.append(arr)

        removed = set(removed)
        
        print(f"Removed Tokens : \n{removed}")

        return result

######################################

############## adress POS of token with symbols ##############


def convert_pos(data, key=".", target_pos="NN"):
    result = []

    for tags in data:
        arr = []
        for tag in tags:
            if key in tag[0] : arr.append((tag[0], target_pos))
            else : arr.append(tag)
        result.append(arr)
    
    return result

######################################

################### POS correction ################

def pos_correction(data, correction_dict):
    result = []

    for tags in data:
        corrected = []
        for token, pos in tags:
            if token in correction_dict.keys() :
                corrected.append((token, correction_dict[token]))
            else : 
                corrected.append((token, pos))
        result.append(corrected)
    
    return result

##############################################

############## lemmatization ##############

class lemmatization_nltk:
    def __init__(self, data, lemmatizer, pos_table, allowed_pos=['noun', 'verb', 'adjective', 'adverb']):
        self.data = data
        self.lemmatizer = lemmatizer
        self.allowed_pos = []
        for pos in allowed_pos:
            if pos == 'noun' : self.allowed_pos.extend(pos_table.Eng_tag[0])
            elif pos == 'verb' : self.allowed_pos.extend(pos_table.Eng_tag[2])
            elif pos == 'adjective' : self.allowed_pos.extend(pos_table.Eng_tag[3])
            elif pos == 'adverb' : self.allowed_pos.extend(pos_table.Eng_tag[4])

    def lemmatize(self):
        result = []

        for tags in self.data:
            arr = []
            for token, pos in tags:
                if pos in self.allowed_pos :
                    pos_info = pos[0].lower()
                    if pos_info == 'j' : pos_info = 'a'
                    elif pos_info =='w' : pos_info = 'r'
                    try : arr.append(self.lemmatizer.lemmatize(token, pos_info))
                    except : print(token, pos, pos_info)
            result.append(arr)
        
        return result

class lemmatization_spacy:
    def __init__(self, data, pos_table, allowed_pos=['noun', 'verb', 'adjective', 'adverb']):

        self.allowed_pos = []
        for pos in allowed_pos:
            if pos == 'noun' : self.allowed_pos.extend(pos_table.Eng_tag[0])
            elif pos == 'verb' : self.allowed_pos.extend(pos_table.Eng_tag[2])
            elif pos == 'adjective' : self.allowed_pos.extend(pos_table.Eng_tag[3])
            elif pos == 'adverb' : self.allowed_pos.extend(pos_table.Eng_tag[4])
        
        self.data = []

        for tags in data:
            tokens = []
            for token, pos in tags : 
                if pos in self.allowed_pos: tokens.append(token)
            self.data.append(" ".join(tokens))

        self.nlp_spacy = spacy.load('en_core_web_sm')
    
    def lemmatize(self, lemma_dict={'saw' : 'see', 'men' : 'man'}):
        result = []

        for text in self.data:
            doc = self.nlp_spacy(text)
            lemmatized_tokens = []
            for token in doc:
                lemma = token.lemma_
                if lemma in lemma_dict.keys() : lemmatized_tokens.append(lemma_dict[lemma])
                else : lemmatized_tokens.append(lemma)
            
            result.append(lemmatized_tokens)
        
        return result

#################################################


def to_pickle(data, file_name, root='./'):
    with open(root+file_name+'.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def to_csv(data, file_name, root='./'):
    df = pd.DataFrame(data)
    df.to_csv(root+file_name+'.csv', index=False)


class check_pos:
    def __init__(self, data):
        self.data = data
        self.dots = {}
        self.apostrophes = {}
        self.dots_wo = {}
        self.apostrophes_wo = {}

        arr_dot = []
        arr_apostrophe = []

        for tags in self.data:
            for tag in tags:
                if "." in tag[0] : arr_dot.append(tag[0])
                elif "'" in tag[0] : arr_apostrophe.append(tag[0])
        
        for dot in set(arr_dot):
            self.dots[dot] = set([tag[1] for tag in merge(self.data) if tag[0] == dot])
        
        for apos in set(arr_apostrophe):
            self.apostrophes[apos] = set([tag[1] for tag in merge(self.data) if tag[0] == apos])
        
        for dot in set(arr_dot):
            removed = nltk.pos_tag([re.sub("[^a-z]", "", dot)])
            self.dots_wo[removed[0][0]] = [removed[0][1]]
        
        for apos in set(arr_apostrophe):
            removed = nltk.pos_tag([re.sub("[^a-z]", "", apos)])
            self.apostrophes_wo[removed[0][0]] = [removed[0][1]]
        
    
    def pos_with_symbol(self):
        print(f"tagged token with apostrophe : \n{self.apostrophes}")
        print(f"tagged token with dot : \n{self.dots}") 

    def pos_without_symbol(self):
        print(f"tagged token without apostrophe : \n{self.apostrophes_wo}")
        print(f"tagged token without dot : \n{self.dots_wo}") 