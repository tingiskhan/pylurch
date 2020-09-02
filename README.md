# pylurch
Lightweight [Falcon](https://falcon.readthedocs.io/en/stable/) application for serving ML models.

## Features
TODO

## Install
You install the library via
```
pip install git+https://github.com/tingiskhan/ml-server
```
There's also a Docker file included for serving the `example` model.

## Usage
TODO
 
## Example
 A really trivial example follows below. 
 
 Start out by initializing the application. There's a Docker compose script
 included, so just navigate to the folder and run

 ```
 docker-compose up -d
 ```
 
 Now, let's train and predict.
 
 ```python
import pandas as pd
import numpy as np
from pylurch.contract.interfaces import GenericModelInterface

# ===== Generate some dummy data ===== #
x = pd.DataFrame(np.random.normal(size=(10000, 10)))
y = (x.sum(axis=1) <= x.mean(axis=1)).astype(np.float32)

# ===== Set up interface, perform fit and predict ===== #
mi = GenericModelInterface('http://localhost:8080/', 'logreg')
mi.fit(x, y=y.to_frame(), session_name='default-classification-model')

yhat = mi.predict(x).iloc[:, 0]
yhat.index = yhat.index.astype(int)

print(f'Precision is: {(yhat.sort_index() == y).mean():.2%}')
 ```
We could repeat the above training and prediction but instead using a neural network by changing the `endpoint` 
parameter of `GenericModelInterface` to `nn`.