from nltk import FreqDist
from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plotWordCloud(data, mode, filepath='./'):

    if mode == 'ST':
        wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
        count = FreqDist(data).most_common(50)
        
        
    elif mode == 'TT':
        wc = WordCloud(font_path='C:\\Users\\marti\\AppData\\Local\\Microsoft\\Windows\\Fonts\\윤고딕330.ttf', 
        width=1000, height=600, background_color="white", random_state=0)
        count = Counter(data).most_common(50)
    
    plt.imshow(wc.generate_from_frequencies(dict(count)))
    plt.axis("off")

    plt.savefig(f'{filepath}WordCloud.png')
    plt.show()