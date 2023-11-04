# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:54:52 2023

@author: MSI
"""

import json
import pandas as pd

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random




def recommendAds(sentiment,target,age):
    if sentiment == "Angry":
        sentiment = "Sad"
    if 20 <= age <= 100:
        age=">20"
    elif 18 <= age < 20:
        age=">18"
    elif 10 <= age < 18:
        age=">10"
    else:
        age="None"
        
    #importing the dataset
    with open('Videosdata.json', encoding='utf-8') as content:
        data1 = json.load(content)
    #getting all the data to lists
    Videopath = []
    Sentiment = []
    Topic = []
    Target = []
    Age = []

    for intent in data1:
        Videopath.append(intent['Video'])
        Sentiment.append(intent['Sentiment'])
        Topic.append(intent['Topic'])
        Target.append(intent['Target'])
        Age.append(intent['Age'])

    #converting to dataframe
    data = pd.DataFrame({"index":[i for i in range(len(Videopath))],
                        "Video":Videopath,
                        "Sentiment":Sentiment,
                        "Topic":Topic,
                        "Target": Target,
                        "Age": Age})


    selected_features = ['Video', 'Sentiment', 'Topic', 'Target', 'Age']
    # merge selected fields 
    merge_selected_fields = data['Video']+' '+data['Sentiment']+' '+data['Topic']+' '+data['Target']+' '+data['Age']

    # converting the text data to feature vectors

    tokenizer = TfidfVectorizer()

    feature_vectors = tokenizer.fit_transform(merge_selected_fields)

    similarity = cosine_similarity(feature_vectors)
    Sentiment_list = data['Sentiment']
    Target_list = data['Target']
    Age_list = data['Age']
    find_close_match_for_sentiment = difflib.get_close_matches(sentiment, Sentiment_list)
    find_close_match_for_target = difflib.get_close_matches(target, Target_list)
    find_close_match_for_age = difflib.get_close_matches(age, Age_list)
    # finding the index of the closest match
    index_close_match = data[(data['Sentiment'] == find_close_match_for_sentiment[0])|(data['Target'] == find_close_match_for_target[0])|(data['Age'] == find_close_match_for_age[0])]['index'].values[0]
    # getting a list of similar ads
    similarity_score = list(enumerate(similarity[index_close_match]))
    # sorting the ads based on their similarity score
    sorted_list = sorted(similarity_score, key=lambda x:x[1], reverse=True)
    Adspath = []
    i = 0

    for ad in sorted_list:
        index = ad[0]
        ad_title = data[data['index'] == index]['Video'].values[0]
        ad_topic = data[data['index'] == index]['Topic'].values[0]
        ad_age = data[data['index'] == index]['Age'].values[0]
        if i<3:
            Adspath.append(ad_title)
            i += 1
    return random.choice(Adspath)