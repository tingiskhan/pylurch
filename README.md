# ml-server
Lightweight Falcon application for serving ML models.

## Background
There's a lot of good resources for learning about how to structure an API for serving ML algorithms. [Medium](https://medium.com/) provides a plethora of different blogs around the subject, but a lot of them are for very specific use cases in order to get something up and running fast.

As such, I decided to share a library with what I've learnt from trying to deploy ML algorithms via APIs (on cloud architecture). In short, this library provides an "easy way" (I think at least) of deploying ML algorithms, abstracting away all the non-algorithm related parts; such as sending/parsing data/keeping track of running models/storing trained models. 

Please note that the solution is something I've developed in my own time and is therefore not entirely bug free nor optimal, as such see it as a complement to the blogs on Medium rather than a production grade ML API.

Some (key) differences between this library and the libraries usually seen on the Medium blogs:
1. Abstracts away the fit/prediction logic for easier deployment of new models. 
2. Provides database model for storing fitted models as binaries.
3. Allows using different backends for serializing the models, e.g. enables easy serialization to [ONNX](https://github.com/onnx/onnx).

## Install
The library utilizes the [falcon](https://falcon.readthedocs.io/en/stable/) framework and utilizes several other libraries such as [marshmallow](https://marshmallow.readthedocs.io/en/stable/) and [SQLAlchemy](https://www.sqlalchemy.org/).
You install the library via
```
pip install git+https://github.com/tingiskhan/ml-server
```
There's also a Docker file included for serving the `example` model.

## Usage
You use the API as you would any REST based API. Every model exposes the five endpoints:
 1. `put`: Corresponds to sklearn's `fit`. Returns a JSON with `model_key`, corresponding to the internal name of the model. **Parameters**:
     1. `x`: Same as in `sklearn`, a `pandas.DataFrame`. Sent as JSON
     2. `y`: Same as in `sklearn`, depending on the model it's either a `pandas.Series` or `pandas.DataFrame`. Sent as JSON.
     3. `orient`: The orientation of the `DataFrame`s, i.e. the parameter `orient` in `pandas.DataFrame.to_json`
     4. `modkwargs`: Any `kwargs` passed to the instantation of the model.
     5. `algkwargs`: Any `kwargs` passed to `fit` method.
     6. `retrain`: Whether to initiate a new training of the model if one already exists.
     7. `name`: Whether to name the data set, thus deriving the internal key from this name instead of the hash of the data.  
 2. `post`: Corresponds to sklearn's `predict`. **Parameters**: corresponds to i., iii. from 1. as well as `model_key`.       
 3. `patch`: Corresponds to updating an existing model using new data. Only applies to a few models in `sklearn`, and as such needs to be overridden by the user. **Parameters**: corresponds to i., ii., and iii. of 1, as well as `model_key`.
 4. `get`: Checks the status of the latest training session specified by `key`. **Parameters**: Only `model_key`.
 
 ## Example
 A really trivial example follows below. Start by serving the model, we'll use port 8080. 
 ```python
from example.app import init_app
from waitress import serve 

if __name__ == '__main__':
    serve(init_app(), port=8080)
 ```
 
 Now, let's train and predict.
 
 ```python
import pandas as pd
import numpy as np
from ml_api.contract.interfaces import GenericModelInterface

# ===== Generate some dummy data ===== #
x = pd.DataFrame(np.random.normal(size=(10000, 10)))
y = (x.sum(axis=1) <= x.mean(axis=1)).astype(np.float32)

# ===== Set up interface, perform fit and predict ===== #
mi = GenericModelInterface('http://localhost:8080/', 'logreg')
mi.fit(x, y.to_frame(), name='logistic-regression-random-data')

yhat = mi.predict(x).iloc[:, 0]
yhat.index = yhat.index.astype(int)

print(f'Precision is: {(yhat.sort_index() == y).mean():.2%}')
 ```
We could also repeat the above training and prediction using a neural network instead by
just changing the `endpoint` parameter of `GenericModelInterface` to `nn`.