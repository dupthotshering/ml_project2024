# transformers.py
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for resizing images
class ResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size=(64, 64)):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        resized_images = [cv2.resize(img, self.size) for img in X]
        return np.array(resized_images)

# Custom transformer for preprocessing images
class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape(len(X), -1) / 255.0
