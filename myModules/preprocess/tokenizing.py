def tokenizing_ST(articles, tokenizer):
    tokenized = []

    for article in articles:
        tokenized.append(tokenizer.tokenize(article))
    
    return tokenized

def tokenizing_TT(articles, tagList, pos='all'):
    if pos == 'noun':
        nouns = []
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[0]:
                    tags.append(str(tag[0]))
            nouns.append(tags)
        return nouns
    
    elif pos == 'verb':
        verb=[]
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[2]:
                    tags.append(str(tag[0]))
            verb.append(tags)
        return verb
    
    elif pos == 'adjective':
        adjective = []
        for article in articles:
            tags = []
            for tag in article:
                if tag[1] in tagList.Kor_tag[3]:
                    tags.append(str(tag[0]))
            adjective.append(tags)
        return adjective
    
    elif pos == 'all':
        all = []
        for article in articles:
            tokens = []
            for tag in article:
                tokens.append(str(tag[0]))
            all.append(tokens)
        return all
    
    else :
        print("Invalid POS mode! must be one of : 'none', 'verb', 'adjective'")
        return -1