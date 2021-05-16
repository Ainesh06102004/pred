import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps


X, y = fetch_openml("mnist_784",version=1, return_X_y=True)
xtrain, xtest, ytrain, ytest = tts(X,y, random_state= 9, train_size = 7500, test_size = 2500 )

xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

lr = LogisticRegression(solver="saga", multi_class ="multinomial").fit(xtrainscaled, ytrain)

def make_prediction(image):
    img =   Image.open(image)
    imgbw = img.convert('L')
    imgbwrsz = imgbw.resize((28,28), Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imgbwrsz, pixelfilter)
    imgbwrszscale = np.clip(imgbwrsz - minpixel, 0, 255)
    maxpixel = np.max(imgbwrsz)
    imgbwrszscale = np.asarray(imgbwrszscale)/maxpixel
    
    testsample = np.array(imgbwrszscale).reshape(1,780)
    pred = lr.predict(testsample)
    return pred[0]