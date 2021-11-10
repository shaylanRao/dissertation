import tweepy
import innit_tweepy

api = innit_tweepy.getTweepyApi()

choice = "open.spotify.com lang:en -filter:retweets"

song_list = []
user_screen_name_list = []
all_tweets = []

tweets = tweepy.Cursor(api.search_tweets, q=choice).items(5)


def get_twitter_song_list():
    for tweet in tweets:
        urls = tweet.entities["urls"]
        try:
            url = urls[0]['expanded_url']
        except IndexError:
            url = 'null'

        if url[0:31] == "https://open.spotify.com/track/":
            song_list.append(url[31:53])
            # print(url[31:53])
            # print("\n")

    return song_list


def get_user_list():
    for tweet in tweets:
        user_screen_name_list.append(tweet.user.screen_name)


def get_all_users_tweets():
    for userID in user_screen_name_list:
        user_tweets = api.user_timeline(screen_name=userID,
                                        count=10,
                                        include_rts=False,
                                        tweet_mode='extended'
                                        )
        time_created = []
        for tweet in user_tweets:
            time_created.append([tweet.created_at, tweet.full_text])

        all_tweets.append([userID, time_created])


def _main_():
    get_user_list()
    get_all_users_tweets()
    # print(all_tweets)


_main_()
