import re
import time
import tweepy
import secrets

def login():
    auth = tweepy.OAuthHandler(secrets.api_key, secrets.api_secret)
    auth.set_access_token(secrets.access_token, secrets.access_secret)
    return tweepy.API(auth, wait_on_rate_limit=True)

def create_tweet(api, message):
    api.update_status(message)

if __name__ == "__main__":
    bot = login()

    checked = []
    for i in range(1, 1000):
        for tweet in bot.search_tweets(q='lgbtq', count=100, lang='en'):
            print(str(tweet.id))
            if tweet.id not in checked:
                with open("./data/tweetdata-exp.txt", "a", encoding='utf-8') as tweet_file:
                    if tweet.text.startswith("RT"):
                        break
                    if tweet.text.startswith("@"):
                        break
                    if tweet.text == "":
                        break
                    if tweet.truncated:
                        print(tweet)
                        tweet_file.write(re.sub(r'https\S+', '', tweet.extended_tweet['full_text'], flags=re.MULTILINE) + "\n")
                    else:
                        tweet_file.write(re.sub(r'https\S+', '', tweet.text, flags=re.MULTILINE) + "\n")
                    checked.append(tweet.id)
            time.sleep(2)