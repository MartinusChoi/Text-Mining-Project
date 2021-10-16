from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plotWordCloud(data, filepath='./'):
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)

    count = FreqDist(data).most_common(50)
    plt.imshow(wc.generate_from_frequencies(dict(count)))
    plt.axis("off")

    plt.savefig(f'{filepath}WordCloud.png')
    plt.show()