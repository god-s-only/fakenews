from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.models import News
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
import uvicorn
import re
import nltk
import pickle
import numpy as np

nltk.download('stopwords')

ann = load_model('my_model.h5', compile=False)
with open("countvectorizer.pkl", 'rb') as f:
    cv = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(req: News):
    news = req.news
    corpus = []
    news = re.sub('[^a-zA-Z]', ' ', news)
    news = news.lower()
    news = news.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if word not in set(all_stopwords)]
    news = ' '.join(news)
    corpus.append(news)
    
    
    news_vector = cv.transform(corpus).toarray()
    
    prediction = ann.predict(news_vector)[0]
    probability_real = prediction[0]  
    probability_fake = 1 - probability_real  

    return {
        "Probability that this news is real is: ": round(probability_real * 100, 2),
        "Probability that this news is fake is: ": round(probability_fake * 100, 2)
    }

if __name__ == '__main__':
    uvicorn.run(app)
