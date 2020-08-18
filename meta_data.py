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
import sys
from collections import Counter

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Example: Accessing the Twitter Streaming API")
    parser.add_argument("--file", "-f", help="Input file")
    parser.add_argument("--credentials", "-c", help="Please input credential file (json format)", required=True)

    args = parser.parse_args()

    ## Load your credentials
    with open(args.credentials) as json_file:
        credentials = json.load(json_file)

    CONSUMER_KEY=credentials["CONSUMER_KEY"]
    CONSUMER_SECRET=credentials["CONSUMER_SECRET"]
    ACCESS_TOKEN=credentials["ACCESS_TOKEN"]
    ACCESS_TOKEN_SECRET=credentials["ACCESS_TOKEN_SECRET"]

    output_file = args.file[:-4].__add__("_meta.tsv")

    #sys.exit()

    file_ = args.file

    api = twitter.Api()

    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN,
                      access_token_secret=ACCESS_TOKEN_SECRET)

    df = pd.read_csv(file_, sep='\t', header = None, names=["Id","Text"])

    twitter_ids = df.Id.values

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
        
        try:
            api = twitter.Api(consumer_key=CONSUMER_KEY,
                            consumer_secret=CONSUMER_SECRET,
                            access_token_key=ACCESS_TOKEN,
                            access_token_secret=ACCESS_TOKEN_SECRET)
            user = api.GetStatus(str(tweet)).AsDict()
            
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

            # If a user has 0 followers
            try:
                meta_dict["followers_count"].append(user["user"]["followers_count"])
            except:
                meta_dict["followers_count"].append(0)

            # If an account doesn't follow any other users
            try:
                meta_dict["following_count"].append(user["user"]["friends_count"])
            except:
                meta_dict["following_count"].append(0)

            try:
                meta_dict["statuses_count"].append(user["user"]["statuses_count"])
            except:
                meta_dict["statuses_count"].append(0)

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

        if (idx+1)%600 == 0:
            print("Sleeping")
            time.sleep(900)
            
    meta_df = pd.DataFrame(meta_dict)

    final_df = pd.merge(df,meta_df, on="Id")

    print("Original df: {}, Meta df: {}, Final df: {}".format(len(df), len(meta_df), len(final_df)))
    print(error_messages)

    final_df.to_csv(output_file, sep="\t", index=False)