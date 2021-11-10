import tweepy

consumer_key = 'UufJ607HmfImJ0lP8aRq5mfsE'
consumer_secret = 'dsERJyMw0x5B5bbjo7XlcMhUWuMdd5BI884NlZA747Kg5wMUzv'
access_token = '1450235258784337923-LPGZuv1f6UVZ61c1huzFz66OjIV0yL'
access_secret = 'qnYQotg4V6veml9nvAtrDcNgk5vtd6RrQU1soZWxt69y6'

tweetsPerQry = 100
maxTweets = 1000000
hashtag = "#mencatatindonesia"

authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
maxId = -1
tweetCount = 0
while tweetCount < maxTweets:
	if(maxId <= 0):
		newTweets = api.search(q=hashtag, count=tweetsPerQry, result_type="recent", tweet_mode="extended")
	else:
		newTweets = api.search(q=hashtag, count=tweetsPerQry, max_id=str(maxId - 1), result_type="recent", tweet_mode="extended")

	if not newTweets:
		print("Tweet Habis")
		break

	for tweet in newTweets:
		print(tweet.full_text.encode('utf-8'))

	tweetCount += len(newTweets)
	maxId = newTweets[-1].id