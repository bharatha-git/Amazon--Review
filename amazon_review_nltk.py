# -*- coding: utf-8 -*-
"""amazon-review-NLTK.ipynb


"""

import io
import nltk
from google.colab import files
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

sid = SentimentIntensityAnalyzer()

uploaded = files.upload()

import pandas as pd

df = pd.read_csv(io.BytesIO(uploaded['amazonreviews.tsv']),sep='\t')

df.head()

df['label'].value_counts()

df.dropna(inplace=True)

blanks = []

for i,lb,rv in df.itertuples():
  if type(rv) == str:
    if rv.isspace():
      blanks.append(i)



df.iloc[0]['review']

sid.polarity_scores(df.iloc[0]['review'])

''' Applying Polarity Scores to every single column in the datasets '''

df['scores'] = df['review'].apply(lambda review : sid.polarity_scores(review))

df.head()

df['compound'] = df['scores'].apply(lambda d:d['compound'])
df['pos'] = df['scores'].apply(lambda d:d['pos'])
df['neg'] = df['scores'].apply(lambda d:d['neg'])
df['neu'] = df['scores'].apply(lambda d:d['neu'])

df.head()

df['comp_score'] = df['compound'].apply(lambda score : 'pos' if score>=0 else 'neg')

df.head()

print('The confusion matrix is\n')
print(confusion_matrix(df['label'],df['comp_score']),'\n')

print('Classification Report is \n')
print(classification_report(df['label'],df['comp_score']),'\n')

print('Accuracy score is \n')
print(accuracy_score(df['label'],df['comp_score']),'\n')

