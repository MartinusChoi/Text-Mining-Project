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
    def calc_properties(self, edge_num, data):
        self.edge_num = edge_num
        self.data = data
        self.max_freq = -1
        self.quartile = 0

        G_centrality = nx.Graph()

        for idx in range(self.edge_num):
            G_centrality.add_edge(self.data['word1'][idx], self.data['word2'][idx], weight=int(self.data['freq'][idx]))
            if self.max_freq < int(self.data['freq'][idx]): self.max_freq = int(self.data['freq'][idx])
        
        self.quartile = self.max_freq / 4
        
        self.dgr = nx.degree_centrality(G_centrality)
        self.cls = nx.closeness_centrality(G_centrality)
        self.btw = nx.betweenness_centrality(G_centrality)
        self.eig = nx.eigenvector_centrality(G_centrality)
        self.pgr = nx.pagerank(G_centrality)

        self.dgr = sorted(self.dgr.items(), key=operator.itemgetter(1), reverse=True)
        self.cls = sorted(self.cls.items(), key=operator.itemgetter(1), reverse=True)
        self.btw = sorted(self.btw.items(), key=operator.itemgetter(1), reverse=True)
        self.eig = sorted(self.eig.items(), key=operator.itemgetter(1), reverse=True)
        self.pgr = sorted(self.pgr.items(), key=operator.itemgetter(1), reverse=True)
    
    def plot(self, centrality='degree', title='Network Web Analysis', root='./'):
        G = nx.Graph()

        # node 추가
        if centrality == 'degree':
            for idx in range(len(self.dgr)):
                G.add_node(self.dgr[idx][0], weight=self.dgr[idx][1])
        elif centrality == 'closeness':
            for idx in range(len(self.cls)):
                G.add_node(self.cls[idx][0], weight=self.cls[idx][1])
        elif centrality == 'betweenness':
            for idx in range(len(self.btw)):
                G.add_node(self.btw[idx][0], weight=self.btw[idx][1])
        elif centrality == 'eigenvector':
            for idx in range(len(self.eig)):
                G.add_node(self.eig[idx][0], weight=self.eig[idx][1])
        elif centrality == 'pagerank':
            for idx in range(len(self.pgr)):
                G.add_node(self.pgr[idx][0], weight=self.pgr[idx][1])
        
        # edge 추가
        for idx in range(self.edge_num):
            G.add_edge(self.data['word1'][idx], self.data['word2'][idx], weight=int(self.data['freq'][idx]))
        
        # 토폴로지 형태 정의
        # pos = graphviz_layout(G, prog='neato')
        pos = graphviz_layout(G, prog='twopi')

        # Network 시각화
        fig = plt.figure(figsize=(30, 20))
        plt.margins(x=0.1, y=0.2)
        ax = fig.gca()

        # figure에 node 그리기
        Blues_modified = cm.get_cmap('Blues', 256)
        newcmp = ListedColormap(Blues_modified(np.linspace(0.2, 0.8, 4)))
        nc = nx.draw_networkx_nodes(G, pos, node_size=[node[1]['weight']*20000 for node in G.nodes(data=True)], \
            node_color=[node[1]['weight'] for node in G.nodes(data=True)], cmap=newcmp,
            node_shape='o', alpha=0.9, linewidths=0.4, edgecolors='#000000')

        # figure에 label 그리기
        nx.draw_networkx_labels(G, pos=pos, font_size=16, alpha=0.7, font_color='black', font_weight='bold')

        # figure에 edge 그리기
        Greys_modified = cm.get_cmap('Greys', 256)
        newcmp = ListedColormap(Greys_modified(np.linspace(0.2, 1.0, 4)))
        ec = nx.draw_networkx_edges(G, pos, edge_color=[edge[2]['weight'] for edge in G.edges(data=True)], \
            edge_cmap=newcmp, style='solid', width=2, connectionstyle='arc3,rad=-0.3')
        
        # title 지정
        plt.title(title, fontsize=25)
        # axis 선 안보이게 설정
        plt.axis('off')

        # color bar 추가
        axins = inset_axes(ax,\
            width='1%',
            height='30%',
            loc='center right',
            borderpad=0)
        cbar = plt.colorbar(nc, cax=axins)
        cbar.ax.set_ylabel('Centrality', rotation=270, fontsize=12, labelpad=15)

        axins = inset_axes(ax, \
            width='1%',
            height='30%',
            loc='center left',
            borderpad=0)
        cbar = plt.colorbar(ec, cax=axins)
        cbar.ax.set_ylabel('Word Pair Freqency', rotation=270, fontsize=12, labelpad=15)

        # figure 저장
        plt.savefig(root+title+'.png')
        # figure 출력
        plt.show()