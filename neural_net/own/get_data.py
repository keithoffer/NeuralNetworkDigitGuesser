import os
import urllib.request

if not os.path.exists('mnist_data'):
    os.mkdir('mnist_data')

# Download data
print("Downloading test data ...")
urllib.request.urlretrieve('http://pjreddie.com/media/files/mnist_test.csv','mnist_data/mnist_test.csv')
print("Downloading train data ...")
urllib.request.urlretrieve('http://pjreddie.com/media/files/mnist_train.csv','mnist_data/mnist_train.csv')
