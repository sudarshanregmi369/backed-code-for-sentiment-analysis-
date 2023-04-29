import re
import tensorflow as tf
import numpy as np
from app.domain.tweet import Tweet, TweetAnalyzed
import nltk
nltk.download('words')

# https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
def clean_tweet(tweet):
    temp = tweet.lower()
    # temp = re.sub("'", "", temp) # to avoid removing contractions in english
    # Removing hashtags and mentions
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    # Removing links
    temp = re.sub(r'http\S+', '', temp)
    # Removing punctuations
    # temp = re.sub('[()!?]', ' ', temp)
    # temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)

    
    words = set(nltk.corpus.words.words())

    sent = temp
    " ".join(w for w in nltk.wordpunct_tokenize(sent) \
            if w.lower() in words or not w.isalpha())
    # 'Io to the beach with my'

    return sent


# Load the saved model
import os
model_path = os.path.join( os.getcwd(),"model.h5")

model = tf.keras.models.load_model(model_path)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')

def analyze_tweet(tweet : Tweet) -> "TweetAnalyzed":
    
        
    # Define the input data for prediction
    text = tweet["text"]
    clean_text = clean_tweet(text)
    new_data = [clean_text]

    # Define the preprocessing steps
    tokenizer.fit_on_texts(new_data)
    sequences = tokenizer.texts_to_sequences(new_data)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post', truncating='post')

    # Use the preprocessed input data to make predictions with the loaded model
    predictions = model.predict(padded_sequences)

    sentiment = ""
    score = 0
    # Print the predictions
    for i, sentence in enumerate(new_data):
        sentiment = 'POSITIVE' if predictions[i] > 0.5 else 'NEGATIVE'
        score = predictions[i][0]
        # print(f"'{sentence}' has a {sentiment} sentiment with a probability of {np.max(predictions[i]):.2f}")
        
    
    # clean_text = clean_tweet(text)
    label = sentiment
    score = score
    # output = pipe(clean_text)[0]
    # label = output.get("label")
    # score = output.get('score')
    analyzed_tweet = TweetAnalyzed(text=text,score=score,label=label)
    
    # print(f"{tweet} analyses")   
    return analyzed_tweet
       