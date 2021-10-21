from nltk import FreqDist
from collections import Counter

import pandas as pd
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def calcTags(tagged, tagList, mode):
    noun = []
    pronoun = []
    verb = []
    adjective = []
    adverb = []
    prepnconj = []
    determiner = []
    interjection = []
    number = []
    foreignW = []
    modal = []
    josa = []
    possesiveS = []
    others = []

    if mode == 'ST':
        tag_lang = 'Eng_tag'
    elif mode == 'TT':
        tag_lang = 'Kor_tag'

    for tag in tqdm(tagged):
        if tag[1] in tagList[tag_lang][0]:
            noun.append(tag)
        elif tag[1] in tagList[tag_lang][1]:
            pronoun.append(tag)
        elif tag[1] in tagList[tag_lang][2]:
            verb.append(tag)
        elif tag[1] in tagList[tag_lang][3]:
            adjective.append(tag)
        elif tag[1] in tagList[tag_lang][4]:
            adverb.append(tag)
        elif tag[1] in tagList[tag_lang][5]:
            prepnconj.append(tag)
        elif tag[1] in tagList[tag_lang][6]:
            determiner.append(tag)
        elif tag[1] in tagList[tag_lang][7]:
            interjection.append(tag)
        elif tag[1] in tagList[tag_lang][8]:
            number.append(tag)
        elif tag[1] in tagList[tag_lang][9]:
            foreignW.append(tag)
        elif tag[1] in tagList[tag_lang][10]:
            modal.append(tag)
        elif tag[1] in tagList[tag_lang][11]:
            josa.append(tag)
        elif tag[1] in tagList[tag_lang][12]:
            possesiveS.append(tag)
        elif tag[1] in tagList[tag_lang][13]:
            others.append(tag)
    
    countDict = {'noun' : len(noun), 'pronoun' : len(pronoun), 'verb' : len(verb), 'adjective' : len(adjective), 'adverb' : len(adverb), \
        'prepnconj' : len(prepnconj), 'determiner' : len(determiner), 'interjection' : len(interjection), 'number' : len(number),
        'foreignW' : len(foreignW), 'modal' : len(modal), 'josa' : len(josa), 'possesiveS' : len(possesiveS), 'others' : len(others), 'total' : len(tagged)}
    
    classDict = {'noun' : noun, 'pronoun' : pronoun, 'verb' : verb, 'adjective' : adjective, 'adverb' : adverb, \
        'prepnconj' : prepnconj, 'determiner' : determiner, 'interjection' : interjection, 'number' : number,
        'foreignW' : foreignW, 'modal' : modal, 'josa' : josa, 'possesiveS' : possesiveS, 'others' : others}
    
    return countDict, classDict

def CountNLabelOverPOS(countDict):
    count = []
    label = []
    for key, val in countDict.items():
        count.append(val)
        label.append(key)
    return count, label

def showCountNRate(countDict, title):
    total = countDict['total']
    print(f"[{title}]")
    print("-----"*5)
    print(f"전체 토큰 수 | {total}개")
    for key, val in countDict.items():
        if key == 'total': break
        pct = round(val / total * 100, 2)
        print(f"[{key}] 토큰 수 : {val}개 | 비율 : {pct}%")
    print("-----"*5)

def tag2csv(countDict, classDict, tagList, filepath, mode):
    df = tagList.copy()

    if mode == 'ST':
        lang = 'Eng'
        df['Eng_tagged'] = [classDict['noun'], classDict['pronoun'], classDict['verb'], classDict['adjective'], \
            classDict['adverb'], classDict['prepnconj'], classDict['determiner'], classDict['interjection'], classDict['number'],
            classDict['foreignW'], classDict['modal'], classDict['josa'], classDict['possesiveS'], classDict['others']]
    elif mode == 'TT':
        lang = 'Kor'
        df['Kor_tagged'] = [classDict['noun'], classDict['pronoun'], classDict['verb'], classDict['adjective'], \
            classDict['adverb'], classDict['prepnconj'], classDict['determiner'], classDict['interjection'], classDict['number'],
            classDict['foreignW'], classDict['modal'], classDict['josa'], [], classDict['others']]
    
    cnts = []
    pcts = []
    total = countDict['total']
    for key, val in countDict.items():
        if key == 'total': break
        pct = round(val / total * 100, 2)
        cnts.append(val)
        pcts.append(pct)
    
    df[lang + '_tagged_count'] = cnts
    df[lang + 'Eng_tagged_%'] = pcts

    df.to_csv(f'{filepath}.csv', index=False)

def plotTagFrequency(data, labels, title, filepath='./'):
    total = data[-1]
    data = data[:-1]
    labels = labels[:-1]

    fig = plt.figure(figsize=(15, 10))
    sns.set_style('dark')
    ax = fig.add_subplot()
    plt.xticks(rotation = -45)

    plot = sns.barplot(x=labels, y=data)

    for i in range(len(data)):
        plot.text(x=i, y=data[i]/2+600, s=data[i], horizontalalignment='center')
        pct = round(data[i]/total * 100, 2)
        plot.text(x=i, y=data[i]/2, s=str(pct) + '%', horizontalalignment='center')

    plt.title(title, fontsize=15)
    plt.savefig(f'{filepath + title}.jpg')
    plt.show()

def plotOverallFrequency(tokenized, mode, filepath='./', num=30, wo=False, period=1):

    if mode == 'ST':
        freq = FreqDist(tokenized)
        freq = freq.most_common(num)
        freq = dict(freq)

        data = list(freq.values())
        labels = []
        for token in freq.keys():
            labels.append(str(token))
    elif mode == 'TT':
        count = Counter(tokenized)
        count = count.most_common(num)
        count = dict(count)

        data = list(count.values())
        labels = []
        for token in count.keys():
            labels.append(str(token))

    if wo is False:
        m = "with StopWords"
    elif wo is True:
        m = "without StopWords"
    
    df = pd.DataFrame()
    df['Term'] = labels
    df['Frequency'] = data
    
    fig = plt.figure(figsize=(15, 10))
    plt.xticks(rotation = -45)

    plot = sns.barplot(x=labels, y=data)

    if mode == 'TT':
        plt.rc('font', family='Malgun Gothic')
    
    title = f"[Period {period}] Frequency [{m}] (most {num})"
    plt.title(title, fontsize=15)
    plt.savefig(f'{filepath + title}.jpg')
    df.to_csv(f'{filepath + title}.csv', index=False)
    plt.show()

def plotPOSFrequency(classDict, mode, filepath='./', pos='noun', wo=False, num=30, period=1):
    
    if mode == 'ST':
        freq = FreqDist(classDict[pos])
        freq = freq.most_common(num)
        freq = dict(freq)

        data = list(freq.values())
        labels = []
        for tag in freq.keys():
            labels.append(str(tag[0]))
    elif mode == 'TT':
        count = Counter(classDict[pos])
        count = count.most_common(num)
        count = dict(count)

        data = list(count.values())
        labels = []
        for tag in count.keys():
            labels.append(str(tag[0]))
    
    if wo is False:
        m = "with StopWords"
    elif wo is True:
        m = "without StopWords"

    df = pd.DataFrame()
    df['Term'] = labels
    df['Frequency'] = data

    fig = plt.figure(figsize=(15, 10))
    plt.xticks(rotation = -45)

    plot = sns.barplot(x=labels, y=data)

    if mode == 'TT':
        plt.rc('font', family='Malgun Gothic')
    
    title = f"[Period {period}] Frequency in '{pos}' [{m}] (most {num})"
    plt.title(title, fontsize=15)
    plt.savefig(f'{filepath + title}.jpg')
    df.to_csv(f'{filepath + title}.csv', index=False)
    plt.show()