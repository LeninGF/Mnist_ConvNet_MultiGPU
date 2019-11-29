import os
from tensorflow._api.v1.keras.models import load_model
from tensorflow._api.v1.keras.datasets import mnist
from tensorflow._api.v1.keras.utils import to_categorical

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
ytest = to_categorical(ytest)
xtest = xtest.reshape(-1,28,28,1)/255.0

dir_root = os.getcwd()
path_model = os.path.join(dir_root, 'models/multi_gpu_model/multigpu.h5')

mnist_model = load_model(filepath=path_model)
print(mnist_model.summary())
loss, acc = mnist_model.evaluate(xtest,ytest,verbose=2)

print('loss: {} acc: {}'.format(loss, acc))

