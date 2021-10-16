from tqdm.notebook import tqdm
import nltk

def tagging(articles):
    tagged = []

    for article in tqdm(articles):
        tagged.append(nltk.pos_tag(article))
    
    return tagged