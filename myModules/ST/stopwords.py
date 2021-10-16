def removeStopWords(articles, stopwords=[], new_stopwords=[]):
    if len(new_stopwords) != 0:
        stopwords.extend(new_stopwords)
    
    removed = []

    for article in articles:
        removed.append([token for token in article if token not in stopwords])
    return removed