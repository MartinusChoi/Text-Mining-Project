def DataLoader(filePathList):
    texts = []

    for filepath in filePathList:
        with open(filepath, 'r') as f:
            text = f.read()
            texts.append(text)
    
    return texts