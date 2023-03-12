import fasttext
import math

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
file = open('/workspace/datasets/fasttext/top_words.txt','r')
lines = file.readlines()

threshold = 0.75

for word in lines:
    synonymsProbPairList = model.get_nearest_neighbors(word, k=10)
    word = word.strip()
    lineToPrint = word
    isNotEmpty = False
    for synonymsProbPair in synonymsProbPairList:
        if synonymsProbPair[0] > threshold and word != synonymsProbPair[1]:
            isNotEmpty = True
            lineToPrint += "," + synonymsProbPair[1]
    if isNotEmpty:
        print(lineToPrint)
        
            
