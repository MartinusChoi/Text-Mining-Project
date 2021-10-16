import pandas as pd
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models

def buildDTM(articles) :
    Dict = corpora.Dictionary(articles)
    corpus = [Dict.doc2bow(article) for article in articles]
    return corpus, Dict

def topicWords(model, num_topic_words):
    topicWords = []
    for topic_id in range(model.num_topics):
        topic_word_probs = model.show_topic(topic_id, num_topic_words)
        for topic_word, prob in topic_word_probs:
            topicWords.append([topic_id, topic_word, prob])
    topicWords = pd.DataFrame(topicWords)
    topicWords.columns = ['topic_id', 'topic_word', 'prob']
    return topicWords

def visualizeLDA(model, Corp, Dict):
    pyLDAvis.enable_notebook()

    return pyLDAvis.gensim_models.prepare(model, Corp, Dict)