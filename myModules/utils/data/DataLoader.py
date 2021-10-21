def DataLoader(filePathList, mode):
    texts = []

    if mode == 'ST':
        for filepath in filePathList:
            with open(filepath, 'r', encoding='latin_1') as f:
                text = f.read()
                texts.append(text)
    elif mode == 'TT':
        for filepath in filePathList:
            with open(filepath, 'r') as f:
                text = f.read()
                texts.append(text)
    
    return texts