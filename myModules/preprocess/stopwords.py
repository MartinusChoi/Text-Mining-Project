from tqdm.notebook import tqdm

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