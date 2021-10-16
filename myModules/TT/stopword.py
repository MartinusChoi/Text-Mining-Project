from tqdm.notebook import tqdm

def removeStopWord(articles, stop_tag_list, Kor_stopwords):
    removed = []
    for article in tqdm(articles):
        arr = []
        for tag in article:
            if (tag[1] not in stop_tag_list) & (tag[0] not in Kor_stopwords) & (len(tag[0]) != 1):
                arr.append(tag)
        removed.append(arr)
    return removed