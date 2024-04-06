from flask import Flask, render_template, request
import re
import joblib

app = Flask(__name__)

# Load the trained model
pipeline = joblib.load('sentiment_analysis_model.joblib')

def preprocess_text(text):
    if isinstance(text, str):  
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''  

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    if request.method == 'POST':
        tweet_content = request.form['tweet_content']
        cleaned_tweet = preprocess_text(tweet_content)
        sentiment = pipeline.predict([cleaned_tweet])[0]
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
