import pandas as pd
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
from itertools import product
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

    return pyLDAvis.gensim_models.prepare(topic_model=model, corpus=Corp, dictionary=Dict, mds='mmds')

class BestLDAPram:
    def __init__(self, data, random_state=42):
        self.data = data
        self.random_state = random_state
        self.corpus, self.dictionary = buildDTM(self.data)
    
    def param_search(self, param_grid):
        param_combi_table = []
        
        param_combination = list(product(param_grid['num_topics'], param_grid['alpha'], param_grid['eta']))
        
        for num_topics, alpha, eta in tqdm(param_combination, desc="Testing Parameter Combination"):
            coherence_value = self.calc_coherence(num_topics, alpha, eta)
            param_combi_table.append([num_topics, alpha, eta, coherence_value])
        
        self.param_combi_table = pd.DataFrame(param_combi_table)
        self.param_combi_table = ['num_topics', 'alpha', 'eta', 'coherence']

        self.best_num_topics = self.get_best_param_value(param='num_topics')
        self.best_alpha = self.get_best_param_value(param='alpha')
        self.best_eta = self.get_best_param_value(param='eta')

        print(f"Best Number of Topics : {self.best_num_topics}")
        print(f"Best Alpha : {self.best_alpha}")
        print(f"Best Number of Topics : {self.best_eta}")
        
    def calc_coherence(self, num_topic, alpha, eta):
        model = models.LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topic, \
            alpha=alpha, eta=eta, random_state=self.random_state)
        
        coherence_model = CoherenceModel(model=model, texts=self.data, dictionary=self.dictionary, coherence='c_v')
        
        return coherence_model.get_coherence()
    
    def get_best_param_value(self, param):
        values = self.param_combi_table[param].unique()
        coherences = []
        for value in values:
            coherences.append(self.param_combi_table.coherence[self.param_combi_table[param] == value].mean())
        idx = np.argmax(coherences)
        best_param = values[idx]

        return best_param
    
    def get_best_params(self):
        return self.best_num_topics, self.best_alpha, self.best_eta

    def get_axis_values(self, param):
        values = self.param_combi_table[param].unique()
        coherences = []
        for value in values:
            coherences.append(self.param_combi_table.coherence[self.param_combi_table[param] == value].mean())
        
        return values, coherences
    
    def plot_coherence_per_topics(self, title='Coherence per Topic Num', root='./'):
        plt.figure()

        num_topics, coherences = self.get_axis_values(param='num_topics')
        
        plt.plot(num_topics, coherences)

        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence')

        plt.title(title)
        plt.savefig(root+title+'.png')
        plt.show()

    def plot_coherence_per_alpha(self, title='Coherence per alpha', root='./'):
        plt.figure()

        alphas, coherences = self.get_axis_values(param='alpha')
        
        plt.plot(alphas, coherences)

        plt.xlabel('Alpha')
        plt.ylabel('Coherence')

        plt.title(title)
        plt.savefig(root+title+'.png')
        plt.show()
    
    def plot_coherence_per_eta(self, title='Coherence per eta', root='./'):
        plt.figure()

        etas, coherences = self.get_axis_values(param='eta')

        plt.plot(etas, coherences)

        plt.xlabel('Eta')
        plt.ylabel('Coherence')

        plt.title(title)
        plt.savefig(root+title+'.png')
        plt.show()