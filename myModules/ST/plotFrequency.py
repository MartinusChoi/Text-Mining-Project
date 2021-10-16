from nltk import FreqDist
import seaborn as sns
import matplotlib.pyplot as plt

def plotOverallFrequency(tokenized, filepath='./', num=30, wo=False, period=1):
    freq = FreqDist(tokenized)
    freq = freq.most_common(num)
    freq = dict(freq)
    freq

    data = list(freq.values())
    labels = []
    for token in freq.keys():
        labels.append(str(token))

    if wo is False:
        m = "with StopWords"
    elif wo is True:
        m = "without StopWords"
    
    fig = plt.figure(figsize=(15, 10))
    plt.xticks(rotation = -45)

    plot = sns.barplot(x=labels, y=data)
    
    title = f"[Period {period}] Overall Frequency (most {num}) [{m}]"
    plt.title(title, fontsize=15)
    plt.savefig(f'{filepath + title}.png')
    plt.show()