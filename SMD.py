from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load the model from disk
nb = pickle.load(open('nbmodel.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('NLP.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = nb.predict(vect)
		print(my_prediction)
	return render_template('NLP.html',prediction_text=my_prediction[0])

if __name__ == '__main__':
	app.run(debug=True)
