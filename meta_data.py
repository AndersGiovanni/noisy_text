"""
This script is collecting user meta data.
Based on tutorial from Barbara Plank: https://github.com/bplank/twitter_api_examples/blob/master/TinyTwitterTutorial.ipynb
"""

import argparse
import json
import twitter
import pandas as pd

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

    tweet = api.GetStatus(tid).AsDict()

    print(tweet["user"])

    #df = pd.read_csv(train_data_path, sep='\t')
    #print(df.head(1))