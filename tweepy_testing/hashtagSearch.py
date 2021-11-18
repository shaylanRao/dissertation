import json
import re
from string import punctuation

import tweepy
import innit_tweepy
import pandas as pd
import numpy as np
from IPython.display import display

api = innit_tweepy.getTweepyApi()

choice = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets'

song_list = []
user_screen_name_list = []
all_tweets = []

# How many different users will be searched
tweets = tweepy.Cursor(api.search_tweets, q=choice, result_type='recent').items(10)


def get_twitter_song_list():
    for tweet in tweets:
        urls = tweet.entities["urls"]
        song_list.append(song_id_in_url(urls))
    return song_list


def song_id_in_url(urls):
    try:
        url = urls[0]['expanded_url']
    except IndexError:
        url = ""

    if url[0:31] == "https://open.spotify.com/track/":
        url = (url[31:53])
        return url

    return ""


def get_user_list():
    for tweet in tweets:
        user_screen_name_list.append(tweet.user.screen_name)


def get_users_spotify_tweets(screen_name):
    query = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets' + " " + screen_name
    spotify_tweets = tweepy.Cursor(api.search_tweets, q=query, result_type='recent').items(10)
    return spotify_tweets


def get_all_users_tweets():
    main_df = pd.DataFrame(np.array([["", "", "", ""]]))
    for userID in user_screen_name_list:
        # Gets the timeline of tweets from each user
        user_tweets = api.user_timeline(screen_name=userID,
                                        count=20,
                                        include_rts=False,
                                        exclude_replies=False,
                                        tweet_mode='extended'
                                        )
        data = np.array([["", "", "", ""]])
        for tweet in user_tweets:
            # Cleans the text from the tweet
            tweet_text = clean_text(tweet.full_text)

            # Gets the song id from the URL if the URL is set to a spotify track
            urls = tweet.entities["urls"]
            song_url = song_id_in_url(urls)

            # Appends the tweet data to past data (from same user)
            data = np.append(data, [[userID, tweet_text, song_url, tweet.created_at]], axis=0)

        # Converts the array into a dataframe
        main_df = main_df.append(pd.DataFrame(data))

    # Creates csv of dataset
    main_df.to_csv("Dataset_moodify_trials.csv")
    display(main_df)


def clean_text(message):
    tweet_text = message.replace('\n', '')
    tweet_text = re.sub("[^0-9a-zA-Z{} ]+".format(punctuation), "", tweet_text)
    return tweet_text


def format_spotify_tweets(tweet):
    tweet_text = clean_text(tweet.text)
    urls = tweet.entities["urls"]
    song_url = song_id_in_url(urls)
    print(tweet_text)
    if song_url:
        print(song_url)
    else:
        print(tweet)
    print("")


def get_size_of_search(tweet_search_items):
    counter = 0
    for i in tweet_search_items:
        counter += 1

    return counter


def count_iterable(i):
    return sum(1 for e in i)


def _main_():
    get_user_list()
    for user in user_screen_name_list:
        # for some reason can't store this and use it in multiple functions/conditionals
        #     s_tweets = get_users_spotify_tweets(user)
        if count_iterable(get_users_spotify_tweets(user)) > 2:
            print("")
            print("--------------------------------")
            print("USER: ", user)
            for tweet in get_users_spotify_tweets(user):
                format_spotify_tweets(tweet)
    # get_all_users_tweets()


_main_()
