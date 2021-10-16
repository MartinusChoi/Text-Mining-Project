def DataLoader(filePathList):
    texts = []

    for filepath in filePathList:
        with open(filepath, 'r', encoding='latin_1') as f:
            text = f.read()
            texts.append(text)
    
    return texts