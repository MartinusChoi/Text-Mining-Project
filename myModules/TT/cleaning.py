import re

def cleanText(text):
    return re.sub('[^A-Za-z가-힣 ]', '', text)

def cleaning(texts):
    cleaned=[]
    for text in texts:
        cleaned.append(cleanText(str(text)))
    return cleaned