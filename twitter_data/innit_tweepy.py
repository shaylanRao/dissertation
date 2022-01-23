import tweepy


def get_tweepy_api():
    api_key = 'UufJ607HmfImJ0lP8aRq5mfsE'
    api_secret_key = 'dsERJyMw0x5B5bbjo7XlcMhUWuMdd5BI884NlZA747Kg5wMUzv'
    access_token = '1450235258784337923-LPGZuv1f6UVZ61c1huzFz66OjIV0yL'
    secret_access_token = 'qnYQotg4V6veml9nvAtrDcNgk5vtd6RrQU1soZWxt69y6'

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, secret_access_token)

    api = tweepy.API(auth)

    return api
