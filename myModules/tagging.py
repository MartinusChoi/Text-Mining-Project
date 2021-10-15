from konlpy.tag import Kkma
from tqdm.notebook import tqdm

kkma = Kkma()

def kkmaTagging(texts):
    tagged = []
    for text in tqdm(texts):
        tagged.append(kkma.pos(text))
    return tagged