# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import re
from nltk.stem import WordNetLemmatizer
LMT = WordNetLemmatizer()

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
    	data = [message]
		mess = re.sub('[^a-zA-Z]', ' ', data).lower().split() 
        mess = [LMT.lemmatize(x) for x in mess if not x in set(stopwords.words('english')) - {'not'}]
        mess = " ".join(mess)	
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)