import os
from tensorflow._api.v1.keras.models import load_model
from tensorflow._api.v1.keras.datasets import mnist
from tensorflow._api.v1.keras.utils import to_categorical

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
ytest = to_categorical(ytest)
xtest = xtest.reshape(-1,28,28,1)/255.0

dir_root = os.getcwd()
path_multigpu_model = os.path.join(dir_root, 'models/multigpu.h5')

mnist_mgpu = load_model(filepath=path_multigpu_model)
print(mnist_mgpu.summary())
loss, acc = mnist_mgpu.evaluate(xtest, ytest, verbose=2)

print('loss: {} acc: {}'.format(loss, acc))

path_single_model = os.path.join(dir_root, 'models/singlegpu.h5')

mnist_sgpu = load_model(filepath=path_single_model)
print(mnist_sgpu.summary())
loss, acc = mnist_sgpu.evaluate(xtest, ytest, verbose=2)

print('loss: {} acc: {}'.format(loss, acc))





