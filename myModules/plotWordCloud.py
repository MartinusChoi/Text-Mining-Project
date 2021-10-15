import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from myModules.analysisTags import merge

def plotWordCloud(data, filepath):
    wc = WordCloud(font_path='C:\\Users\\marti\\AppData\\Local\\Microsoft\\Windows\\Fonts\\윤고딕330.ttf',
    width=1000, height=600, background_color="white", random_state=0)

    count = Counter(merge(data)).most_common(50)
    plt.imshow(wc.generate_from_frequencies(dict(count)))
    plt.axis("off")

    plt.savefig(f'{filepath}.png')
    plt.show()