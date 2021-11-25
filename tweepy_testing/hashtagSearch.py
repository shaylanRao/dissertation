import json
import re
from datetime import timedelta
from string import punctuation

import tweepy
import innit_tweepy
import pandas as pd
import numpy as np
from IPython.display import display

from sentiment_testing.sentimentAnalyser import get_senti
from spotipy_testing.graphPlaylist import graph_one_playlist

api = innit_tweepy.getTweepyApi()

choice = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets'
song_list = []
user_screen_name_list = []
all_tweets = []

column_names = ["user_name", "text", "track_id", "tweet_id", "time"]
all_s_tweets = pd.DataFrame(columns=column_names)

num_users = 10
max_song_tweets = 20
num_before_tweets = 3

# Gets recent tweets which include spotify links,  .items(n) -> how many different users will be searched
recent_s_tweets = tweepy.Cursor(api.search_tweets, q=choice, result_type='recent').items(num_users)


def song_id_in_url(urls):
    try:
        url = urls[0]['expanded_url']
    except IndexError:
        url = ""

    if url[0:31] == "https://open.spotify.com/track/":
        url = (url[31:53])
        return url

    return ""


# Adds to pre-defined list of users, the usernames from tweets
def get_user_list():
    for tweet in recent_s_tweets:
        user_screen_name_list.append(tweet.user.screen_name)


# Gets only spotify tweets from a user - passed as string
def get_users_spotify_tweets(screen_name):
    query = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets' + " " + screen_name
    # Gets specified number of tweets that include songs in the tweets
    spotify_tweets = tweepy.Cursor(api.search_tweets, q=query, result_type='recent').items(max_song_tweets)
    return spotify_tweets


# Not in use!
def get_all_users_tweets():
    main_df = pd.DataFrame(np.array([["", "", "", ""]]))
    for userID in user_screen_name_list:
        # Gets the timeline of tweets from each user
        query = 'lang:en exclude:replies -filter:retweets' + " " + userID
        user_tweets = tweepy.Cursor(api.search_tweets, q=query, result_type='recent').items(10)

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


# Cleans text data - passed as string
def clean_text(message):
    # Removes space, flattens text
    tweet_text = message.replace('\n', '')

    # Removes urls
    tweet_text = re.sub(r'http\S+', '', tweet_text)

    # Removes any 'special' characters
    tweet_text = re.sub("[^0-9a-zA-Z{} ]+".format(punctuation), "", tweet_text)
    return tweet_text


def get_s_tweet_data(tweet):
    tweet_text = clean_text(tweet.text)
    urls = tweet.entities["urls"]
    song_url = song_id_in_url(urls)
    return tweet_text, song_url


def get_size_of_search(tweet_search_items):
    counter = 0
    for i in tweet_search_items:
        counter += 1

    return counter


def count_iterable(i):
    return sum(1 for e in i)


def tabulate_s_tweets(user_name, text, track_id, tweet_id, time):
    df = {'user_name': user_name, 'text': text, 'track_id': track_id, 'tweet_id': tweet_id, 'time': time}
    global all_s_tweets
    all_s_tweets = all_s_tweets.append(df, ignore_index=True)
    return None


def create_song_lists():
    all_user_lists = []
    # iterates through each user within s_tweets
    for user in all_s_tweets['user_name'].unique():
        user_song_list = []
        for row in all_s_tweets[all_s_tweets['user_name'] == user].iterrows():
            # Gets track ID from tweet
            if row[1][2] != "":
                user_song_list.append(row[1][2])

        if user_song_list:
            all_user_lists.append(user_song_list)
    return all_user_lists


def get_before_s_tweets():
    example_user = all_s_tweets.iloc[0]
    print("user_name:   ", example_user['user_name'])

    for s_tweet in all_s_tweets[all_s_tweets['user_name'] == example_user['user_name']].iterrows():
        # Gets date (YYY-MM-DD) of tweet - use to limit tweets only going back 7 days - only to keep tweets within bound
        until_date = example_user['time'].date()
        until_date += timedelta(days=1)

        # Gets tweet_id
        tweet_id = s_tweet[1][3]
        messages = ""

        print("Track ID: ", s_tweet[1][2])

        # Query for tweets from user
        query = 'lang:en exclude:replies -filter:retweets ' + example_user['user_name']

        # Gets (upto) 3 tweets from user - until: searches tweets BEFORE given date
        before_s_tweet = tweepy.Cursor(api.search_tweets,
                                       q=query,
                                       result_type='recent',
                                       max_id=tweet_id,
                                       until=until_date
                                       ).items(num_before_tweets)
        for tweet in before_s_tweet:
            if tweet.id != tweet_id:
                if song_id_in_url(tweet.entities["urls"]) == "":
                    messages = '\n'.join([messages, clean_text(tweet.text)])
                    # print("SENTI: ", get_senti(clean_text(tweet.text)))
                else:
                    break
            else:
                messages = '\n'.join([messages, clean_text(tweet.text)])
                # print("FIRST SENTI: ", get_senti(clean_text(tweet.text)))

        # Gets overall sentiment from pas tweets (song  relevant)
        print(get_senti(messages))


def _main_():
    # Creates list of users who have posted using a spotify link in their tweet
    get_user_list()
    for user in user_screen_name_list:
        # for some reason can't store this and use it in multiple functions/conditionals
        #     s_tweets = get_users_spotify_tweets(user)

        # If there are more than 2 tweets that the user has made which includes a spotify track, (DIS-COUNTS USERS WITH LESS - hence not always selected number of users shown in table
        if count_iterable(get_users_spotify_tweets(user)) > 2:

            # For each tweet, extract each component and collate it in a dataframe
            for tweet in get_users_spotify_tweets(user):
                text, song_id = get_s_tweet_data(tweet)
                tabulate_s_tweets(user_name=user, text=text, track_id=song_id, tweet_id=tweet.id, time=tweet.created_at)

    # Displays whole table of all users and corresponding spotify tweets
    # display(all_s_tweets)

    # Saves the tweets related to a track as a csv file [user id, text, track (if there), time]
    all_s_tweets.to_csv("s_tweets_trial.csv")

    # Gets a couple of previous tweets from a user before they posted a specific song
    # get_before_s_tweets()

    # Gets a list of songs
    all_song_lists = create_song_lists()

    # Gets the largest list of songs
    max_list = max(x for x in all_song_lists)
    print(all_song_lists)

    # Graphs the largest song list
    graph_one_playlist(max_list)

    # Outdated
    # Gets all tweets from a user
    # get_all_users_tweets()


_main_()
