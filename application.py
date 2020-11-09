import json
from flask import Flask,request,jsonify

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np



app = Flask(__name__)

max_length = 120

model=load_model("model/MODEL_POC.h5")
token=pickle.load(open("model/token.pkl","rb"))
integer_mapping_set=pickle.load(open("model/integer_mapping_set.pkl","rb")) 

def get_encoded(x): 
  x= str(x).lower()
  x = token.texts_to_sequences([x])
  x = pad_sequences(x, maxlen=max_length, padding = 'post')
  return x
 
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
  
@app.route('/')
def hello():    
    return "APP is running!"

@app.route('/getpath')
def getpath():
    q=request.args['q']    

    x=q
    result_vector=np.argmax(model.predict(get_encoded(x)), axis=-1)
    for key, val in integer_mapping_set.items(): 
      if val == result_vector:
        print(key)
        result=key
   
    print(q)
    output = json.dumps(result, default=set_default)
    return output



if __name__=='__main__':
    app.run() 




