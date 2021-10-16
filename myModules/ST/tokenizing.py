def tokenizing(articles, tokenizer):
    tokenized = []

    for article in articles:
        tokenized.append(tokenizer.tokenize(article))
    
    return tokenized