from flask import Flask,render_template,request
import pickle

app = Flask('__name__')

model = pickle.load(open('spam_model.pkl','rb'))
vectorizer = pickle.load(open('spam_vectorizer.pkl ','rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    message = request.form ['message']
    data    = [message]

    vect = vectorizer.transform(data)
    prediction = model.predict(vect)

    return render_template('index.html',prediction = prediction[0])


if __name__ == "__main__":
    app.run(debug=True)