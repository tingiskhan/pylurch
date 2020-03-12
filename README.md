# ml-server
Light weight application for serving ML models.

## Background
There's a lot of good resources for learning about how to structure an API for serving ML algorithms. [Medium](https://medium.com/) provides a plethora of different blogs around the subject, but a lot of them are for very specific use cases and are often very basic in order to get something up and running fast. 

As such, I decided to develop a library from what I've learnt from trying to deploy ML algorithms via APIs (on cloud architecture). Please note that the solution is something I've devloped in my own time, and is therfore not entirely bug free/optimally strucutured, as such see it as a complement to the blogs on Medium rather than a production grade ML server.

Some (key) differences between this library and the libraries usually seen on the Medium blogs:
1. Abstracts away the fit/prediction logic for easier deployment of new models. 
2. Provides database model for storing fitted models as binaries with meta data instead of files.
3. Allows using different backends for serializing the models, e.g. enables easy access to ONNX.

## Install
While you could install the library, the installation is more aimed at providing a simple way of deploying using Docker. But, to install simply
```
pip install git+https://github.com/tingiskhan/ml-server
```

## Usage
You use the API as you would any REST based API. Every model exposes the five endpoints:
 1. `put`: Corresponds to sklearn's `fit`. Here, you pass the necessary data via the variables `x` and `y`, together with some specific key worded arguments passed to the model. Returns the key the model receives, if you'de like to save the model using a specific name you can pass the `name` variable. Uses `flask-executor` to allow for long running training.
 2. `post`: Corresponds to sklearn's `predict`. As above, you need to pass the data to predict for, which corresponds to the variable `x`. You also need to include the key of the model, returned by `1.`
 3. `patch`: Corresponds to updating the model using data. Only applies to a few models in `sklearn`, and as such needs to be overridden by the user.
 4. `delete`: Deletes all instances of a model.
 5. `get`: Checks the status of the model, i.e. is it still training or can we use it for prediction.
