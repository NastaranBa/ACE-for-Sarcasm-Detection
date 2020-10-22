

import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
import pandas as pd 

dataset = pd.read_csv("../input/test1.csv")

dat= dataset['SIT'].str.split().apply(lambda x: ','.join(list(set(x))))
print(dat)

text= dataset['content']
X=text.values.tolist()


Emotion = dataset['Field1']
emo=Emotion.values.tolist()

corpus = []
for i in range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'[@%*~#+á\xc3\xa1\-\.\']','',review)
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r"\d"," ",review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', str(X[i]))
    corpus.append(review)    
    
words=[]
tokenized=[]
for i in range(len(corpus)):
    words= nltk.word_tokenize(corpus[i])
    tokenized.append(words)

df = pd.DataFrame(list(zip(tokenized, emo)), 
               columns =['Text', 'Emotion']) 


from gensim import corpora, models, similarities
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('..input/embeddings.bin.gz', binary=True) 

l=df['Text']

l


#workbook = xlsxwriter.Workbook('NastaranResults.xlsx')
#worksheet = workbook.add_worksheet()

#fw = open('result.txt', 'w')
# seed_words = open('C:/Users/nasba/Downloads/seedwords1.csv', 'r')
# next(seed_words)


# row=0
# score=0
import csv
with open('embeddings_res.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    for sentence in l:
        #print(sentence)
        avg_all = []

        #for each word of the sentence
        for word in sentence:

            if word in model.vocab:
                seed_words = open('..input/seedwords1.csv', 'r')
                #next(seed_words)

                #print(word)
                avgs = []

                #for each line in the topics
                for line in seed_words:
                    line1 = line.rstrip().split(',')
                    #print(str(line1) + "----------")
                    sim =[]
                    for i in range(0, len(line1)):
                        #print(line1[i])
                        value = model.similarity(word, line1[i])
                        sim.append(value)


                    sum = 0
                    for j in sim:
                        sum += j


                    #print(sum)
                    avg = sum / len(sim)
                    #print(avg)

                    #This is the vector of 5 category
                    avgs.append(avg)

                avg_all.append(avgs)

        if avg_all == []:
            continue
        
        #print(avg_all)

        avg_cols = []

        for j in range(0, len(avg_all[0])):
            col = 0
            for array in avg_all:
                col += array[j]

            avg_cols.append(col / len(avg_all))
        #print(avg_cols, avg_cols.index(max(avg_cols)))
        line = [avg_cols, avg_cols.index(max(avg_cols))]
        writer.writerow(line)
csvFile.close()        
#fw.close()   



import numpy as np
from sklearn.metrics import precision_recall_fscore_support

pred = pd.read_csv('..input/embeddings_res.csv',names=['AVG', 'Emotions'])

pred

pred.Emotions.replace([0,1,2,3,4], ['joy','sadness','anger','fear','disgust'], inplace=True)

from sklearn.metrics import classification_report
y_pred = pred.Emotions
y_true = df.Emotion
target_names = ['joy','sadness','anger','fear','disgust']
print(classification_report(y_true, y_pred, target_names=target_names,average='weighted'))

print(classification_report(y_true, y_pred, target_names=target_names))
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

precision_recall_fscore_support(y_true, y_pred, average='weighted')

precision_recall_fscore_support(y_true, y_pred, average='micro')

precision_recall_fscore_support(y_true, y_pred, average='macro')

from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_true,y_pred)))









