"""Client to Mongo DB."""
import os
import random
from pymongo import MongoClient

MONGODB_HOST = '192.168.1.128'
MONGODB_PORT = 27017
MONGODB_NAME = 'efacstel'


class MongoBaseClient:
    """Client for mongo db."""

    def __init__(self):
        """Init."""
        client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = client[MONGODB_NAME]

    def article_collection(self):
        """Return artile Collection.

        colume:
            'message': text,
            'md5': md5hash,
            'article_layer': int,
            'negative': int,
            'positive': int,
            'long_text': boolean,
            'marked': boolean,
            'fid': string,
        """
        return self.db.article

    def sentiment_dictionary_collection(self):
        """Return positive_dic Collection.

        colume:
            'word': text,
            'sentiment': int,  # 1 = pos, 0 = neg.
            'is_sentiment': boolean,  # If the word is sentiment word.
            'marked': boolean,
        """
        return self.db.sentiment_dictionary

    def article_test(self):
        """Testing article.

         colume:
            'message': text,
            'md5': md5hash,
            'article_layer': int,
            'negative': int,
            'positive': int,
            'long_text': boolean,
            'marked': boolean,
            'fid': string,
        """
        return self.db.article_test


client = MongoBaseClient()
client = client.article_collection()
obj = client.find({"long_text": False, "marked": True})

negative = [5, 4]
positive = [5, 4]

negative_list = []
positive_list = []

print("處理資料中")
for o in obj:
    if o['negative'] in negative:
        negative_list.append(o['message'])
    if o['positive'] in negative:
        positive_list.append(o['message'])

print("{}, {}".format(len(negative_list), len(positive_list)))
print("輸出檔案")

negative_file = "data/n/jasmine_n.txt"
positive_file = "data/p/jasmine_p.txt"
file_cppunt = 16000
if os.path.exists(negative_file):
    os.remove(negative_file)

if os.path.exists(positive_file):
    os.remove(positive_file)

random.shuffle(negative_list)
with open(negative_file, 'a') as file:
    for sentence in negative_list[0:file_cppunt]:
        file.write(sentence+"\n")
random.shuffle(positive_list)
with open(positive_file, 'a') as file:
    for sentence in positive_list[0:file_cppunt]:
        file.write(sentence+"\n")

print(client.find({"long_text": False, "marked": False})[51])
