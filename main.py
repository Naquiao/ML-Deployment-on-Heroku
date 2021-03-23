import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#Creating the instance of the class
app = Flask(__name__)

@app.route("/")
@app.route('/home')
def home():
    return flask.render_template("home.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("model/model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        try:
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list)
            if int(result)==0:
                prediction='Iris-Setosa'
            elif int(result)==1:
                prediction='Iris-Virginica'
            elif int(result)==2:
                prediction='Iris-Versicolour'
            else:
                prediction=f'{int(result)} Not-Defined'
        except ValueError:
            prediction='Data Format Error'

        return render_template("result.html", prediction=prediction)

if __name__=="__main__":

    app.run(port=5001)
