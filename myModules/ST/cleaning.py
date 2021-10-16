import re

def lowerCase(text):
    return text.lower()


def cleanText(text):
    text = lowerCase(text)
    return re.sub('[^a-z ]', '', text)

def cleaning(texts):
    cleaned=[]
    for text in texts:
        cleaned.append(cleanText(str(text)))
    return cleaned