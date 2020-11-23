# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
LMT = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'Reasturent_Sentiment_Multinominal_Model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('Reasturent_Sentiment_Tfidf.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')
	
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
		mess = [LMT.lemmatize(x) for x in message.lower().split() if not x in set(stopwords.words('english')) - {'not'}]
		mess = " ".join(mess)
    	data = [mess]			
    	vect = cv.transform(mess).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)