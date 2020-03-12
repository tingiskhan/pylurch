# ml-server
Lightweight application for serving ML models.

## Background
There's a lot of good resources for learning about how to structure an API for serving ML algorithms. [Medium](https://medium.com/) provides a plethora of different blogs around the subject, but a lot of them are for very specific use cases and are often very basic in order to get something up and running fast. 

As such, I decided to develop a library from what I've learnt from trying to deploy ML algorithms via APIs (on cloud architecture). Please note that the solution is something I've devloped in my own time, and is therfore not entirely bug free/optimally strucutured, as such see it as a complement to the blogs on Medium rather than a production grade ML server.

Some (key) differences between this library and the libraries usually seen on the Medium blogs:
1. Abstracts away the fit/prediction logic for easier deployment of new models. 
2. Provides database model for storing fitted models as binaries with meta data instead of files.
3. Allows using different backends for serializing the models, e.g. enables easy access to ONNX.

## Install
While you could install the library, the setup process is more aimed at providing a simple way of deploying using Docker. But, to install simply
```
pip install git+https://github.com/tingiskhan/ml-server
```

## Usage
You use the API as you would any REST based API. Every model exposes the five endpoints:
 1. `put`: Corresponds to sklearn's `fit`. Returns a JSON with `model-key`, corresponding to the internal name of the model. **Parameters**:
     1. `x`: Same as in `sklearn`, a `pandas.DataFrame`. Sent as JSON
     2. `y`: Same as in `sklearn`, depending on the model it's either a `pandas.Series` or `pandas.DataFrame`. Sent as JSON.
     3. `orient`: The orientation of the `DataFrame`s, i.e. the parameter `orient` in `pandas.DataFrame.to_json`
     4. `modkwargs`: Any `kwargs` passed to the instantation of the model.
     5. `algkwargs`: Any `kwargs` passed to `fit` method.
     6. `retrain`: Whether to initiate a new training of the model if one already exists.
     7. `name`: Whether to name the model, thus deriving the model key from the name instead of the data. If no name is passed, the data is hashed and the resulting key is used for identification.     
 2. `post`: Corresponds to sklearn's `predict`. **Parameters**: corresponds to i., iii. from 1. as well as `model-key`.       
 3. `patch`: Corresponds to updating the model using data. Only applies to a few models in `sklearn`, and as such needs to be overridden by the user. Parameters correspond to i., ii., and iii. of 1.
 4. `delete`: Deletes all instances of a model. **Parameters**: Only `model-key` is required.
 5. `get`: Checks the status of the model, i.e. is it still training or can we use it for prediction. **Parameters**: Only `model-key`.
