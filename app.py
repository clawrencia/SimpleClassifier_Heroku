from flask import Flask, render_template, request
import pickle

#name of the current python module
app = Flask(__name__)

classifier = pickle.load(open('mnb_sentiment_classifier.pkl','rb'))
cv = pickle.load(open('count-Vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method =='POST':
        message = request.form['message']
        data = [message]
        vectorizer = cv.transform(data).toarray()
        prediction_res = classifier.predict(vectorizer)
        return render_template('result.html', prediction=prediction_res)

if __name__ == '__main__': #on running python app.py
    app.run(debug=True)   #run flask app