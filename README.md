# ml-server
Lightweight application for serving ML models.

## Background
There's a lot of good resources for learning about how to structure an API for serving ML algorithms. [Medium](https://medium.com/) provides a plethora of different blogs around the subject, but a lot of them are for very specific use cases and are often rudimentary, in order to get something up and running fast. 

As such, I decided to share a library with what I've learnt from trying to deploy ML algorithms via APIs (on cloud architecture). Please note that the solution is something I've developed in my own time and is therefore not entirely bug free nor optimal, as such see it as a complement to the blogs on Medium rather than a production grade ML API.

Some (key) differences between this library and the libraries usually seen on the Medium blogs:
1. Abstracts away the fit/prediction logic for easier deployment of new models. 
2. Provides database model for storing fitted models as binaries with meta data instead of files. But is possible to store as files.
3. Allows using different backends for serializing the models, e.g. enables easy access to ONNX.
4. Provides a simple database model for managing users, as well as some simple authentication.

## Install
While you could install the library, the setup process is more aimed at providing a simple way of deploying using Docker. But, to install simply
```
pip install git+https://github.com/tingiskhan/ml-server
```
There's also a Docker file included.

## Usage
You use the API as you would any REST based API. Every model exposes the five endpoints:
 1. `put`: Corresponds to sklearn's `fit`. Returns a JSON with `model-key`, corresponding to the internal name of the model. **Parameters**:
     1. `x`: Same as in `sklearn`, a `pandas.DataFrame`. Sent as JSON
     2. `y`: Same as in `sklearn`, depending on the model it's either a `pandas.Series` or `pandas.DataFrame`. Sent as JSON.
     3. `orient`: The orientation of the `DataFrame`s, i.e. the parameter `orient` in `pandas.DataFrame.to_json`
     4. `modkwargs`: Any `kwargs` passed to the instantation of the model.
     5. `algkwargs`: Any `kwargs` passed to `fit` method.
     6. `retrain`: Whether to initiate a new training of the model if one already exists.
     7. `name`: Whether to name the data set, thus deriving the internal key from this name instead of the hash of the data.  
 2. `post`: Corresponds to sklearn's `predict`. **Parameters**: corresponds to i., iii. from 1. as well as `model-key`.       
 3. `patch`: Corresponds to updating an existing model using new data. Only applies to a few models in `sklearn`, and as such needs to be overridden by the user. **Parameters**: corresponds to i., ii., and iii. of 1, as well as `model-key`.
 4. `delete`: Deletes all instances of a model. **Parameters**: Only `model-key` is required.
 5. `get`: Checks the status of the model, i.e. is it still training or can we use it for prediction. **Parameters**: Only `model-key`.
 
 ## Example
 A really trivial example follows below. It's assumed that you have started the server locally on port 5000, which is done as 
 ```python
 from ml_server.app import app


if __name__ == '__main__':
    app.run()
 ```
 
 Now, let's train and predict.
 
 ```python
import pandas as pd
from requests import post, put
import json

address = 'http://localhost:5000/'

headers = {
    'Content-type': 'application/json'
}

# ===== Generate some dummy data ===== #
x = pd.DataFrame(pd.np.random.normal(size=(10000, 10)))
y = (x.sum(axis=1) <= x.mean(axis=1)).astype(pd.np.float32)

# ===== Define parameters to send ===== #
orient = 'columns'

params = {
    'x': x.to_json(orient=orient),
    'y': y.to_frame().to_json(orient=orient),
    'orient': orient
}

# ===== Train the model (logistic regression) ====== #
train = put(address + 'logreg', headers=headers, json=params)
resp = json.loads(train.text)

# ===== Let it train ===== #
sleep(10)

# ===== Predict in sample ===== #
pred = {
    'x': x.to_json(orient=orient),
    'orient': orient,
    'model-key': resp['model-key']
}

predict = post(address + 'logreg', headers=headers, json=pred)
res = json.loads(predict.text)
yhat = pd.read_json(res['y']).iloc[:, 0]

print(f'Precision is: {(yhat == y).mean():.2%}')
 ```
