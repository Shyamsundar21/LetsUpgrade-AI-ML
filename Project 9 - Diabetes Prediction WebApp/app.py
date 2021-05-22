import numpy as np
from flask import Flask, request,  render_template
import pickle

app = Flask(__name__)
with open('model_rf.pkl','rb') as file:
    model = pickle.load(file)
# model = pickle.load(open('model_rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

#     output = round(prediction[0], 2)
    if prediction == 0:
        output = 'You Don\'t have Diabetes'
    else:
        output = 'You have Diabetes'

    return render_template('index.html', prediction_text= output)



if __name__ == "__main__":
    app.run(debug=True)