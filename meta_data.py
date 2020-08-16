"""
This script is collecting user meta data.
Based on tutorial from Barbara Plank: https://github.com/bplank/twitter_api_examples/blob/master/TinyTwitterTutorial.ipynb
"""

import argparse
import json
import twitter
import pandas as pd
import time
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Example: Accessing the Twitter Streaming API")
    #parser.add_argument('--credentials','-c', help="user credentials (required)",required=True)
    parser.add_argument('--count', help="number of tweets", default=10)

    args = parser.parse_args()

    ## Load your credentials
    credentials = { 
            "CONSUMER_KEY": "Bxa9UfGV4K8BGN2aBPCGNFycM",
            "CONSUMER_SECRET": "2OjnpCajULYmNhcWBpzTpURJoRouwZZJlNWXoLLbA1u48DyFzN",
            "ACCESS_TOKEN": "1593686437-rr9z6o1P7rjJVcT5blLN28rWpntJkDLxFRJaYgC",
            "ACCESS_TOKEN_SECRET": "GeplRnQMX5crruq7NkyAhoPGorAniprluiqyFxLSENMp6",
    }

    CONSUMER_KEY=credentials["CONSUMER_KEY"]
    CONSUMER_SECRET=credentials["CONSUMER_SECRET"]
    ACCESS_TOKEN=credentials["ACCESS_TOKEN"]
    ACCESS_TOKEN_SECRET=credentials["ACCESS_TOKEN_SECRET"]

    tid = "1241490299215634434"

    train_data_path = "data/train_lower_entities.tsv"

    api = twitter.Api()

    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN,
                      access_token_secret=ACCESS_TOKEN_SECRET)

    df = pd.read_csv(train_data_path, sep='\t')

    #print(df.head())

    twitter_ids = df.Id.values

    #print(str(df["Text"].loc[df['Id'] == 1242799513619664896]))

    meta_dict = {
            "Id":list(),
            "screen_name":list(),
            "description":list(),
            "favourites_count":list(),
            "followers_count":list(),
            "following_count":list(),
            "statuses_count":list()
    }

    error_messages = list()


    for idx, tweet in tqdm(enumerate(twitter_ids)):
        if idx % 500:
                time.sleep(900)
                print("Sleeping")
        
        try:
            #print(tweet)
            user = api.GetStatus(str(tweet)).AsDict()
            #print(user["user"])
            
            meta_dict['Id'].append(tweet)
            meta_dict["screen_name"].append(user["user"]["screen_name"])

            # Empty description
            try:
                meta_dict["description"].append(user["user"]["description"])
            except:
                meta_dict["description"].append(" ")
            
            # If a user doens't have any favourites
            try:
                meta_dict["favourites_count"].append(user["user"]["favourites_count"])
            except:
                meta_dict["favourites_count"].append(0)

            meta_dict["followers_count"].append(user["user"]["followers_count"])
            
            # If an account doesn't follow any other users
            try:
                meta_dict["following_count"].append(user["user"]["friends_count"])
            except:
                meta_dict["following_count"].append(0)

            #meta_dict["statuses_count"].append(user["user"]["statuses_count"])

        # If tweet id has changed/deleted
        except Exception as e:
            meta_dict['Id'].append(tweet)
            meta_dict["screen_name"].append("__UNK_Screen_Name__")
            meta_dict["description"].append("__UNK_Description__")
            meta_dict["favourites_count"].append(0)
            meta_dict["followers_count"].append(0)
            meta_dict["following_count"].append(0)
            meta_dict["statuses_count"].append(0)
            error_messages.append(e)             
            print(e)
            


    #tweet = api.GetStatus(tid).AsDict()

    #for key, value in tweet.items():
    #        print("{}: {}".format(key, value))

    #df = pd.read_csv(train_data_path, sep='\t')
    #print(df.head(1))