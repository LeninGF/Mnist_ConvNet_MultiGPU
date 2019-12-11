from tensorflow._api.v1.keras.datasets import mnist
from tensorflow._api.v1.keras.utils import to_categorical
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.layers import Dense, Conv2D, Flatten
from tensorflow._api.v1.keras.models import load_model
from tensorflow._api.v1.keras.optimizers import RMSprop
from tensorflow._api.v1.keras.utils import multi_gpu_model
import os
import time
import pathlib

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

xtrain = xtrain.reshape(-1, 28, 28, 1)/255.0
xtest = xtest.reshape(-1, 28, 28, 1)/255.0

model = Sequential()
model.add(Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model_multiple = multi_gpu_model(model, gpus=2, cpu_relocation=True)
model_multiple.compile(optimizer=RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])

print('Training on Multiple GPU ...')
time_start = time.time()
history = model_multiple.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=20)
time_end = time.time()
path_to_save_multi_gpu = os.path.join(os.getcwd(), 'models/multi_gpu_model', 'multigpu.h5')
pathlib.Path(path_to_save_multi_gpu).mkdir(parents=True, exist_ok=True)
model_multiple.save(filepath=path_to_save_multi_gpu)
print('Model training took {} seconds \nModel saved to {}'.format(time_end-time_start, path_to_save_multi_gpu))

print('Training on Single GPU')
time_start = time.time()
history_single = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=20)
time_end = time.time()
path_to_save_single_model = os.path.join(os.getcwd(), 'models/single_model', 'single.h5')
pathlib.Path(path_to_save_single_model).mkdir(parents=True, exist_ok=True)
model.save(filepath=path_to_save_single_model)
print('Model training took {} seconds \nModel saved to {}'.format(time_end-time_start, path_to_save_single_model))

# read_multi_model = load_model(filepath=path_to_save_multi_gpu)
# read_multi_model.summary()
# print('multi gpu model opened')
# read_multi_model.compile(optimizer=RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
# loss, acc = read_multi_model.evaluate(xtest, ytest, verbose=2)

print('training finished')
# read_single_model = load_model(filepath=path_to_save_single_model)
# read_single_model.summary()
# print('single model opened')
# read_single_model.compile(optimizer=RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['acc'])
# loss, acc = read_single_model.evaluate(xtest,ytest,verbose=2)
#

