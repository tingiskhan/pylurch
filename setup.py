from setuptools import setup, find_packages
from ml_server import __version__


setup(
    name='ml_server',
    version=__version__,
    author='Victor Gruselius',
    author_email='victor.gruselius@gmail.com',
    description='API for serving machine learning models',
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-executor',
        'flask-bcrypt',
        'flask-restful',
        'flask-sqlalchemy',
        'flask-httpauth',
        'pandas',
        'gunicorn',
        'pandas',
        'numpy',
        'scikit-learn',
        'onnx',
        'onnxruntime',
        'skl2onnx',
        'dill'
    ]
)