# Visualization
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.notebook import tqdm
import pandas as pd
import operator
import numpy as np
from nltk import bigrams, ConditionalFreqDist, ngrams
import matplotlib.font_manager as fm
from matplotlib import rc
import re

class vocaDict:
    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.count = {}
    
    def getIdOrAdd(self, word):
        if word in self.word2id:
            return self.word2id[word]
        self.word2id[word] = len(self.word2id)
        self.id2word.append(word)
        return len(self.word2id) - 1
    
    def getWord(self, id):
        return self.id2word[id]
    
    def calcWordPairFreq(self, text):
        wordIds = [self.getIdOrAdd(word) for word in text]
        for idx, a in enumerate(tqdm(wordIds, desc="Word Pair Frequency ")):
            for b in wordIds[idx+1:]:
                if a == b: continue
                if a > b: a, b = b, a
                self.count[a, b] = self.count.get((a, b), 0) + 1
        return self.count
    
    def to_csv(self, file_name, root='./'):
        df = []
        for key, val in self.count.items():
            df.append([self.getWord(key[0]), self.getWord(key[1]), val])
        df = pd.DataFrame(df, columns=['word1', 'word2', 'freq'])
        df = df.sort_values(by=['freq'], ascending=False)
        df = df.reset_index(drop=True)

        df.to_csv(root+'word-pair-freq-'+file_name+'.csv', index=False)

####################################################

################## Network #########################


class Network:
    def __init__(self, data, num_node=30, keywords=[]):
        bgrams = [bigrams(tokens) for tokens in data]

        token = []
        for bgram in bgrams:
            token += ([x for x in bgram])
        
        cfd = ConditionalFreqDist(token)
        freq_mat = []

        for i in tqdm(cfd.keys(), desc="Generate Frequency Dist"):
            temp = []
            for j in cfd.keys():
                temp.append(cfd[i][j])
            freq_mat.append(temp)
        freq_mat = np.array(freq_mat)
        freq_df = pd.DataFrame(freq_mat, index=cfd.keys(), columns=cfd.keys())

        if len(keywords) == 0:
            co_occurrence_dict = {}
            for idx in range(len(freq_df)):
                co_occurrence_dict[str(idx)] = freq_df.iloc[idx, :].max()
            sorted_co_occurrence_dict = sorted(co_occurrence_dict.items(), key = lambda item: item[1], reverse=True)
            top_N = sorted_co_occurrence_dict[:num_node]
            top_N_idx = []
            for idx, _ in top_N:
                top_N_idx.append(int(idx))
        
            self.G = nx.from_pandas_adjacency(freq_df.iloc[top_N_idx, :])
        else : 
            valid_idx = keywords.copy()

            tmp = freq_df.loc[keywords, :]

            for idx in keywords:
                valid_idx.extend(tmp.loc[idx, tmp.loc[idx, :] > 0].index)

            co_occurrence_dict = {}
            for idx in valid_idx:
                co_occurrence_dict[idx] = freq_df.loc[idx, :].max()
            sorted_co_occurrence_dict = sorted(co_occurrence_dict.items(), key = lambda item: item[1], reverse=True)
            top_N = sorted_co_occurrence_dict[:num_node]
            top_N_idx = []
            for idx, _ in top_N:
                top_N_idx.append(idx)

            self.G = nx.from_pandas_adjacency(freq_df.loc[top_N_idx, :])

        dgr = nx.degree_centrality(self.G)
        self.dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)

        cls = nx.closeness_centrality(self.G)
        self.cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)

        btw = nx.betweenness_centrality(self.G)
        self.btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)

        eig = nx.eigenvector_centrality(self.G)
        self.eig = sorted(eig.items(), key=operator.itemgetter(1), reverse=True)
    
    def save_centrality_table(self, table_len=10, title='centrality_table', root='./'):
        dgr_df = pd.DataFrame(self.dgr)
        cls_df = pd.DataFrame(self.cls)
        btw_df = pd.DataFrame(self.btw)
        eig_df = pd.DataFrame(self.eig)
        centrality_df = pd.concat([dgr_df, cls_df, btw_df, eig_df], axis=1)
        centrality_df.columns = ['Word', 'Degree Centrality', 'Word', 'Closeness Centrality', \
            'Word', 'Betweenness Centrality', 'Word', 'Eigenvector Centrality']
        centrality_df[:table_len].to_csv(root+title+'.csv')
        return centrality_df[:table_len]
    
    def plot(self, centrality='degree', title='Network-Analysis', root='./'):
        if centrality == 'degree':
            title = title + '-Degree'
            centrality_value = self.dgr
        elif centrality == 'closeness':
            title = title + '-Closeness'
            centrality_value = self.cls
        elif centrality == 'betweenness':
            title = title + '-Betweenness'
            centrality_value = self.btw
        elif centrality == 'eigenvector':
            title = title + '-Eigenvector'
            centrality_value = self.eig

        for token, central in centrality_value:
            self.G.nodes[token]['weight'] = central

        fig = plt.figure(figsize=(30, 20))
        plt.margins(x=0.1, y=0.2)
        plt.rc('font', family='Malgun Gothic')
        font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        ax = fig.gca()

        # edge??? node??? label ????????????
        node_labels = dict((token, token) for token, _ in self.G.nodes(data=True))
        try :
            pos = graphviz_layout(self.G, prog='neato')
        except : 
            pos = nx.layout.fruchterman_reingold_layout(self.G)
            # pos = nx.shell_layout(self.G)
            # pos = nx.kamada_kawai_layout(self.G)
            # pos = nx.random_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, font_family=font_name, labels=node_labels)

        # Node??? centrality??? ?????? weight ??????
        Blues_modified = cm.get_cmap('Blues', 256)
        newcmp = ListedColormap(Blues_modified(np.linspace(0.2, 0.8, 4)))
        nc = nx.draw_networkx_nodes(self.G, pos, 
            node_size=[node[1]["weight"]*20000 for node in self.G.nodes(data=True)],
            node_color=[node[1]['weight'] for node in self.G.nodes(data=True)], 
            cmap=newcmp, node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # Edge??? co-occurrence??? ?????? weight ??????
        Greys_modified = cm.get_cmap('Greys', 256)
        newcmp = ListedColormap(Greys_modified(np.linspace(0.2, 1.0, 4)))
        ec = nx.draw_networkx_edges(self.G, pos, 
            edge_color=[edge[2]['weight'] for edge in self.G.edges(data=True)],
            edge_cmap=newcmp, style='solid', width=2)

        # Title ??????
        plt.title(title, fontsize=25)
        plt.axis('off')

        axins = inset_axes(ax,
                    width='1%',
                    height='30%',
                    loc='center right',
                    borderpad=0)
        cbar = plt.colorbar(nc, cax=axins)
        cbar.ax.set_ylabel('Centrality', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax,
            width='1%',
            height='30%',
            loc='center left',
            borderpad=0)
        cbar = plt.colorbar(ec, cax=axins)
        cbar.ax.set_ylabel('Word Pair Freqency', rotation=270, fontsize=12, labelpad=15)

        plt.savefig(root+title+'.png')
        plt.show()

###############################################################
################# Network Ver 2. ###############################

class Network_v2:
    def __init__(self, data, keywords=['america', 'soviet']):
        # token??? bigram??????
        bgrams = [bigrams(tokens) for tokens in data]

        # bigram ????????? ???????????? list ????????? ??????
        token = []
        for bgram in bgrams:
            token += ([x for x in bgram])

        # ConditionalFreqDist ???????????? ???????????? bigram?????? ?????? frequency dist ??????
        cfd = ConditionalFreqDist(token)
        freq_mat = []

        # frequency dist ????????? Pandas dataframe?????? ??????
        for i in tqdm(cfd.keys(), desc="Generate Frequency Dist"):
            temp = []
            for j in cfd.keys():
                temp.append(cfd[i][j])
            freq_mat.append(temp)
        freq_mat = np.array(freq_mat)
        self.freq_df = pd.DataFrame(freq_mat, index=cfd.keys(), columns=cfd.keys())

        # keywords??? ???????????? Network??? ????????? row?????? index????????? ?????? copy
        valid_idx = keywords.copy()

        # keywords??? ???????????? row?????? freq_dist?????? ??????
        tmp = self.freq_df.loc[keywords, :]

        # ??????????????? ?????? keywords?????? freq ?????? ?????????????????? token?????? valid idx??? ??????
        for idx in keywords:
            valid_idx.extend(tmp.loc[idx, tmp.loc[idx, :] > 0].index)

        relation_list = valid_idx.copy()

        try : relation_list.remove(keywords[1])
        except : pass

        try : relation_list.remove(keywords[0])
        except : pass

        relation_map = [(keywords[0], keywords[0]), (keywords[1], keywords[1])]

        for row_idx in relation_list:
            if self.freq_df[row_idx][keywords[0]] > self.freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[0]))
            elif self.freq_df[row_idx][keywords[0]] < self.freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[1]))
            else : 
                relation_map.append((row_idx, 'neutral'))


        relation_with_america = {}
        relation_with_soviet = {}
        for idx, relation in relation_map:
            if relation == keywords[0]:
                relation_with_america[idx] = self.freq_df.loc[idx, :].max()
            elif relation == keywords[1]:
                relation_with_soviet[idx] = self.freq_df.loc[idx, :].max()

        sorted_relation_with_america = sorted(relation_with_america.items(), key = lambda item : item[1], reverse=True)
        sorted_relation_with_soviet = sorted(relation_with_soviet.items(), key = lambda item : item[1], reverse=True)

        try : top_N_america = sorted_relation_with_america[:10]
        except : top_N_america = sorted_relation_with_america[:]

        try : top_N_soviet = sorted_relation_with_soviet[:10]
        except : top_N_soviet = sorted_relation_with_soviet[:]

        top_N = top_N_america.copy()
        top_N.extend(top_N_soviet)
        top_N = sorted(top_N, key= lambda item : item[1], reverse=True)

        self.top_N_idx = []

        for idx, _ in top_N:
            self.top_N_idx.append(idx)
        
        relation_list = self.top_N_idx.copy()

        try : relation_list.remove(keywords[1])
        except : pass

        try : relation_list.remove(keywords[0])
        except : pass

        self.relation_map = [(keywords[0], keywords[0]), (keywords[1], keywords[1])]

        for row_idx in relation_list:
            if self.freq_df[row_idx][keywords[0]] > self.freq_df[row_idx][keywords[1]]:
                self.relation_map.append((row_idx, keywords[0]))
            elif self.freq_df[row_idx][keywords[0]] < self.freq_df[row_idx][keywords[1]]:
                self.relation_map.append((row_idx, keywords[1]))
            else : 
                self.relation_map.append((row_idx, 'neutral'))

    def remove(self, token):
        return re.sub("[^a-z'\ ]", "", token)

    def remove_dot(self, data):
        result = []

        for tokens in data:
            temp = []
            for token in tokens:
                if "." in token : temp.append(self.remove(token))
                else : temp.append(token)
            result.append(temp)
        
        return result
    
    def generate_network(self):
        self.G = nx.from_pandas_adjacency(self.freq_df.loc[self.top_N_idx, :])

        dgr = nx.degree_centrality(self.G)
        self.dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)

        cls = nx.closeness_centrality(self.G)
        self.cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)

        btw = nx.betweenness_centrality(self.G)
        self.btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)

        eig = nx.eigenvector_centrality(self.G)
        self.eig = sorted(eig.items(), key=operator.itemgetter(1), reverse=True)
    
    def plot(self, centrality='degree', title='Semantic Network Analysis', root='./'):
        if centrality == 'degree':
            title = title + ' (Degree)'
            centrality_value = self.dgr
        elif centrality == 'closeness':
            title = title + ' (Closeness)'
            centrality_value = self.cls
        elif centrality == 'betweenness':
            title = title + ' (Betweenness)'
            centrality_value = self.btw
        elif centrality == 'eigenvector':
            title = title + ' (Eigenvector)'
            centrality_value = self.eig

        for token, central in centrality_value:
            self.G.nodes[token]['weight'] = central

        for token, relation_val in self.relation_map:
            self.G.nodes[token]['relation'] = relation_val

        fig = plt.figure(figsize=(30, 20))
        plt.margins(x=0.1, y=0.2)
        plt.rc('font', family='Malgun Gothic')
        font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        ax = fig.gca()

        # edge??? node??? label ????????????
        node_labels = dict((token, token) for token, _ in self.G.nodes(data=True))
        try :
            pos = graphviz_layout(self.G, prog='neato')
        except : 
            pos = nx.layout.fruchterman_reingold_layout(self.G)
            # pos = nx.shell_layout(self.G)
            # pos = nx.kamada_kawai_layout(self.G)
            # pos = nx.random_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, font_family=font_name, labels=node_labels)

        # Node??? centrality??? ?????? weight ??????
        Blues_modified = cm.get_cmap('Blues', 256)
        bluescmp = ListedColormap(Blues_modified(np.linspace(0.2, 0.8, 4)))
        nc_blues = nx.draw_networkx_nodes(self.G, pos,
            nodelist = [node[0] for node in self.G.nodes(data=True) if node[1]['relation'] == 'america'], 
            node_size=[node[1]["weight"]*20000 for node in self.G.nodes(data=True) if node[1]['relation'] == 'america'],
            node_color=[node[1]['weight'] for node in self.G.nodes(data=True) if node[1]['relation'] == 'america'], 
            cmap=bluescmp, node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # Node??? centrality??? ?????? weight ??????
        Reds_modified = cm.get_cmap('Reds', 256)
        redscmp = ListedColormap(Reds_modified(np.linspace(0.2, 0.8, 4)))
        nc_reds = nx.draw_networkx_nodes(self.G, pos,
            nodelist = [node[0] for node in self.G.nodes(data=True) if node[1]['relation'] == 'soviet'], 
            node_size=[node[1]["weight"]*20000 for node in self.G.nodes(data=True) if node[1]['relation'] == 'soviet'],
            node_color=[node[1]['weight'] for node in self.G.nodes(data=True) if node[1]['relation'] == 'soviet'], 
            cmap=redscmp, node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # Edge??? co-occurrence??? ?????? weight ??????
        Greys_modified = cm.get_cmap('Greys', 256)
        newcmp = ListedColormap(Greys_modified(np.linspace(0.2, 1.0, 4)))
        ec = nx.draw_networkx_edges(self.G, pos, 
            edge_color=[edge[2]['weight'] for edge in self.G.edges(data=True)],
            edge_cmap=newcmp, style='solid', width=2)

        # Title ??????
        plt.title(title, fontsize=25)
        plt.axis('off')

        axins = inset_axes(ax,
                    width='1%',
                    height='30%',
                    loc='center right',
                    borderpad=0)
        cbar = plt.colorbar(nc_blues, cax=axins)
        cbar.ax.set_ylabel('Centrality (America)', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax,
                    width='1%',
                    height='30%',
                    loc='upper right',
                    borderpad=0)
        cbar = plt.colorbar(nc_reds, cax=axins)
        cbar.ax.set_ylabel('Centrality (Soviet)', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax,
            width='1%',
            height='30%',
            loc='lower right',
            borderpad=0)
        cbar = plt.colorbar(ec, cax=axins)
        cbar.ax.set_ylabel('Word Pair Freqency', rotation=270, fontsize=12, labelpad=15)

        plt.savefig(root+title+'.png')
        plt.show()

    def save_centrality_table(self, table_len=10, title='centrality_table', root='./'):
        dgr_df = pd.DataFrame(self.dgr)
        cls_df = pd.DataFrame(self.cls)
        btw_df = pd.DataFrame(self.btw)
        eig_df = pd.DataFrame(self.eig)
        centrality_df = pd.concat([dgr_df, cls_df, btw_df, eig_df], axis=1)
        centrality_df.columns = ['Word', 'Degree Centrality', 'Word', 'Closeness Centrality', \
            'Word', 'Betweenness Centrality', 'Word', 'Eigenvector Centrality']
        centrality_df[:table_len].to_csv(root+title+'.csv')
        return centrality_df[:table_len]

########################################################
########## Network version 3 ########################

class Network_v3:
    def __init__(self, data, max_node=10, keywords=['america', 'soviet']):
        self.keywords = keywords

        # ngram??? list ????????? ??????
        token = self.to_ngram(data=data, n=2)

        # frequency dist ????????? Pandas dataframe?????? ??????
        self.freq_df = self.gen_frequency_df(token)

        related_with_america, related_with_soviet = self.gen_relation_list(freq_df=self.freq_df, keywords=keywords)

        try : top_N_america = related_with_america[:max_node]
        except : top_N_america = related_with_america[:]

        try : top_N_soviet = related_with_soviet[:max_node]
        except : top_N_soviet = related_with_soviet[:]

        top_N = top_N_america.copy()
        top_N.extend(top_N_soviet)
        top_N = sorted(top_N, key= lambda item : item[1], reverse=True)

        self.top_N_idx = []

        for idx, _ in top_N:
            self.top_N_idx.append(idx)
        
        self.relation_map = self.gen_relation_map(freq_df=self.freq_df, top_N_idx=self.top_N_idx, keywords=keywords)
    
    def to_ngram(self, data, n=2):
        # token??? bigram??????
        bgrams = [ngrams(tokens, n) for tokens in data]

        # bigram ????????? ???????????? list ????????? ??????
        token = []
        for bgram in bgrams:
            token += ([x for x in bgram])
        
        return token
    
    def gen_frequency_df(self, ngrams):
        # ConditionalFreqDist ???????????? ???????????? bigram?????? ?????? frequency dist ??????
        cfd = ConditionalFreqDist(ngrams)
        freq_mat = []

        # frequency dist ????????? Pandas dataframe?????? ??????
        for i in tqdm(cfd.keys(), desc="Generate Frequency Dist"):
            temp = []
            for j in cfd.keys():
                temp.append(cfd[i][j])
            freq_mat.append(temp)
        freq_mat = np.array(freq_mat)
        freq_df = pd.DataFrame(freq_mat, index=cfd.keys(), columns=cfd.keys())

        return freq_df
    
    def gen_relation_list(self, freq_df, keywords):
        # keywords??? ???????????? Network??? ????????? row?????? index????????? ?????? copy
        relation_list = keywords.copy()

        # keywords??? ???????????? row?????? freq_dist?????? ??????
        tmp = freq_df.loc[keywords, :]

        # ??????????????? ?????? keywords?????? freq ?????? ?????????????????? column idx?????? valid idx??? ??????
        for idx in keywords:
            relation_list.extend(tmp.loc[idx, tmp.loc[idx, :] > 0].index)

        # ?????? keyword??? ????????? token?????? list?????? ??????
        relation_list = [val for val in relation_list if val != keywords[0]]
        relation_list = [val for val in relation_list if val != keywords[1]]

        # tuple??? ????????? ?????? : ?????? token
        # tuple??? ????????? ?????? : ????????? keyword
        relation_map = [(keywords[0], keywords[0]), (keywords[1], keywords[1])]

        # ????????? keyword ??? ??? ?????? ?????? keyword??? ???????????? tuple ????????? append
        for row_idx in relation_list:
            if self.freq_df[row_idx][keywords[0]] > self.freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[0]))
            elif self.freq_df[row_idx][keywords[0]] < self.freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[1]))
            else : # ??? keyword ??? ??????(?????? ?????? ??????)?????? ?????? ??????
                relation_map.append((row_idx, 'neutral'))

        relate_with_america = {}
        relate_with_soviet = {}
        for idx, relation in relation_map:
            if relation == keywords[0]:
                relate_with_america[idx] = freq_df.loc[idx, :].max()
            elif relation == keywords[1]:
                relate_with_soviet[idx] = freq_df.loc[idx, :].max()

        sorted_relate_with_america = sorted(relate_with_america.items(), key = lambda item : item[1], reverse=True)
        sorted_relate_with_soviet = sorted(relate_with_soviet.items(), key = lambda item : item[1], reverse=True)

        return sorted_relate_with_america, sorted_relate_with_soviet
    
    def gen_relation_map(self, freq_df, top_N_idx, keywords):
        relation_list = top_N_idx.copy()

        relation_list = [val for val in relation_list if val != keywords[0]]
        relation_list = [val for val in relation_list if val != keywords[1]]

        relation_map = [(keywords[0], keywords[0]), (keywords[1], keywords[1])]

        for row_idx in relation_list:
            if freq_df[row_idx][keywords[0]] > freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[0]))
            elif freq_df[row_idx][keywords[0]] < freq_df[row_idx][keywords[1]]:
                relation_map.append((row_idx, keywords[1]))
            else : 
                relation_map.append((row_idx, 'neutral'))
        
        return relation_map

    def remove(self, token):
        return re.sub("[^a-z'\ ]", "", token)

    def remove_dot(self, data):
        result = []

        for tokens in data:
            temp = []
            for token in tokens:
                if "." in token : temp.append(self.remove(token))
                else : temp.append(token)
            result.append(temp)
        
        return result
    
    def generate_network(self):
        self.G = nx.from_pandas_adjacency(self.freq_df.loc[self.top_N_idx, :])

        dgr = nx.degree_centrality(self.G)
        self.dgr = sorted(dgr.items(), key=operator.itemgetter(1), reverse=True)

        cls = nx.closeness_centrality(self.G)
        self.cls = sorted(cls.items(), key=operator.itemgetter(1), reverse=True)

        btw = nx.betweenness_centrality(self.G)
        self.btw = sorted(btw.items(), key=operator.itemgetter(1), reverse=True)

        eig = nx.eigenvector_centrality(self.G)
        self.eig = sorted(eig.items(), key=operator.itemgetter(1), reverse=True)
    
    def plot(self, centrality='degree', title='Semantic Network Analysis', root='./'):
        if centrality == 'degree':
            title = title + ' (Degree)'
            centrality_value = self.dgr
        elif centrality == 'closeness':
            title = title + ' (Closeness)'
            centrality_value = self.cls
        elif centrality == 'betweenness':
            title = title + ' (Betweenness)'
            centrality_value = self.btw
        elif centrality == 'eigenvector':
            title = title + ' (Eigenvector)'
            centrality_value = self.eig

        for token, central in centrality_value:
            self.G.nodes[token]['weight'] = central

        for token, relation_val in self.relation_map:
            self.G.nodes[token]['relation'] = relation_val

        fig = plt.figure(figsize=(30, 20))
        plt.margins(x=0.1, y=0.2)
        plt.rc('font', family='Malgun Gothic')
        font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
        ax = fig.gca()

        # edge??? node??? label ????????????
        node_labels = dict((token, token) for token, _ in self.G.nodes(data=True))
        try :
            pos = graphviz_layout(self.G, prog='neato')
        except : 
            pos = nx.layout.fruchterman_reingold_layout(self.G)
            # pos = nx.shell_layout(self.G)
            # pos = nx.kamada_kawai_layout(self.G)
            # pos = nx.random_layout(self.G)
        nx.draw_networkx_labels(self.G, pos, font_family=font_name, labels=node_labels)

        # Node??? centrality??? ?????? weight ??????
        Blues_modified = cm.get_cmap('Blues', 256)
        bluescmp = ListedColormap(Blues_modified(np.linspace(0.2, 0.8, 4)))
        nc_blues = nx.draw_networkx_nodes(self.G, pos,
            nodelist = [node[0] for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[0]], 
            node_size=[node[1]["weight"]*20000 for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[0]],
            node_color=[node[1]['weight'] for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[0]], 
            cmap=bluescmp, node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # Node??? centrality??? ?????? weight ??????
        Reds_modified = cm.get_cmap('Reds', 256)
        redscmp = ListedColormap(Reds_modified(np.linspace(0.2, 0.8, 4)))
        nc_reds = nx.draw_networkx_nodes(self.G, pos,
            nodelist = [node[0] for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[1]], 
            node_size=[node[1]["weight"]*20000 for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[1]],
            node_color=[node[1]['weight'] for node in self.G.nodes(data=True) if node[1]['relation'] == self.keywords[1]], 
            cmap=redscmp, node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # Edge??? co-occurrence??? ?????? weight ??????
        Greys_modified = cm.get_cmap('Greys', 256)
        newcmp = ListedColormap(Greys_modified(np.linspace(0.2, 1.0, 4)))
        ec = nx.draw_networkx_edges(self.G, pos, 
            edge_color=[edge[2]['weight'] for edge in self.G.edges(data=True)],
            edge_cmap=newcmp, style='solid', width=2)

        # Title ??????
        plt.title(title, fontsize=25)
        plt.axis('off')

        axins = inset_axes(ax,
                    width='1%',
                    height='30%',
                    loc='center right',
                    borderpad=0)
        cbar = plt.colorbar(nc_blues, cax=axins)
        cbar.ax.set_ylabel('Centrality (America)', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax,
                    width='1%',
                    height='30%',
                    loc='upper right',
                    borderpad=0)
        cbar = plt.colorbar(nc_reds, cax=axins)
        cbar.ax.set_ylabel('Centrality (Soviet)', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax,
            width='1%',
            height='30%',
            loc='lower right',
            borderpad=0)
        cbar = plt.colorbar(ec, cax=axins)
        cbar.ax.set_ylabel('Word Pair Freqency', rotation=270, fontsize=12, labelpad=15)

        plt.savefig(root+title+'.png')
        plt.show()

    def save_centrality_table(self, table_len=10, title='centrality_table', root='./'):
        dgr_df = pd.DataFrame(self.dgr)
        cls_df = pd.DataFrame(self.cls)
        btw_df = pd.DataFrame(self.btw)
        eig_df = pd.DataFrame(self.eig)
        centrality_df = pd.concat([dgr_df, cls_df, btw_df, eig_df], axis=1)
        centrality_df.columns = ['Word', 'Degree Centrality', 'Word', 'Closeness Centrality', \
            'Word', 'Betweenness Centrality', 'Word', 'Eigenvector Centrality']
        centrality_df[:table_len].to_csv(root+title+'.csv')
        return centrality_df[:table_len]