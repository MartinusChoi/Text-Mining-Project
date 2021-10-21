import re

def lowerCase(text):
    return text.lower()

def cleanST(text):
    text = lowerCase(text)
    return re.sub('[^a-z ]', '', text)

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