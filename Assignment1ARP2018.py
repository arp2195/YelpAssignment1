import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from nltk import sent_tokenize,word_tokenize
import random

df = pd.read_csv('yelp.csv')
df = df[['stars','text']]


class sentiment:
    def __init__(self,df):
        self.df = pd.DataFrame(df)
    @staticmethod
    def rating(df,col1,new):
        grade = []
        for i in df[col1]:
            if i >=4:
                grade.append('POS')
            else:
                grade.append('NEG')
            
        df[new] = grade
        return df
    def apply_rate(self):
        self.df = sentiment.rating(self.df,'stars','rating')
        return self.df

    def run_progs(self):
        self.apply_rate()
        
new = sentiment(df)

new.run_progs()
analysis = new.df

pos = analysis[analysis['rating']=='POS']
neg = analysis[analysis['rating']=='NEG']

def tupling(df,sentiment,col):
   train_tags = []
   for words in df[col][:10000]:
       train_tags.append(tuple([sentiment,words]))
   return train_tags

positive = tupling(pos,'pos','text')
negative = tupling(neg,'neg','text')
training = []

for i in positive:
    training.append(i)
    
for i in negative:
    training.append(i)

random.seed(123)
random.shuffle(training)

cl = NaiveBayesClassifier(training)


p1 = pos[10001:15001]
p2 = neg[10001:15001]

large_set = pd.concat([p1,p2])


testing = []

for i in large_set['text']:
    testing.append(i)

    
fin_sent = []   
for i in testing:
    blob = TextBlob(i,classifier=cl)
    for s in blob.sentiment:
        fin_sent.append(s)
    

df_pp = pd.DataFrame(fin_sent)

file = pd.merge(large_set,df_pp,left_index=True,right_index=True)
file = file.rename(columns={0:'Confidence'})


new_col = []
for i in file['Confidence']:
    if i >.5:
        new_col.append('POS')
    else:
        new_col.append('NEG')
        
file['Sentiemnt_Grade']=new_col

print(file.head())