import os 

def makeFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)