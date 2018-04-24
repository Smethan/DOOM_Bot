# from alexnet import AlexNet as googlenet
from KerasModel import KMod as googlenet
from random import shuffle
import numpy as np
import gc
from keras.utils import np_utils
import random


garbagecount = 1

FILE_END = 137
WIDTH = 4
HEIGHT = 25
LR = 1e-3
EPOCHS = 30
MODEL_NAME='DOOM_Beta_Bot'
PREVIOUS_MODEL ='DOOM_Beta_Bot'
LOAD_MODEL = True
GARBAGE_LOGGING = False
if GARBAGE_LOGGING:
    file = open('gc_log.txt', 'w')

model = googlenet(WIDTH, HEIGHT, output=14)

# if LOAD_MODEL:
#     model.load(PREVIOUS_MODEL)
#     print('loaded previous model!')
# train_data = np.load('training_data-1.npy')

for e in range(EPOCHS):
    data_order = [i for i in range(1,FILE_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        try:

            file_name = 'training_data-{}.npy'.format(i)
            train_data = np.load(file_name)
            print('Loading training_data-{}.npy'.format(i))
            train = train_data[:-50]
            # test = train_data[-50:]

            X = np.array([i[0] for i in train])
            Y = np.array([i[1] for i in train])


            TrainData = np.random.choice([0,1], size=(6300,1,4,25))
            Y = Y.reshape(6300,)
            Y = np_utils.to_categorical(Y, 14)


            model.fit(TrainData,Y, epochs=5, verbose=1)


            if GARBAGE_LOGGING:
                d = {}

                for o in gc.get_objects():
                    name = type(o).__name__
                    if name not in d:
                        d[name] = 1
                    else:
                        d[name] += 1

                items = sorted(d.items(),key=lambda x:x[1])

                file.write('\n')
                file.write('GARBAGE LOG NUMBER {}\n'.format(garbagecount))
                file.write('--------------------------------\n')
                for key, value in items:


                    file.write(key + ' ' + str(value) + '\n')
                    #print(key, value)
                garbagecount += 1

            # if count%10 == 0:
            #     print('SAVING MODEL!')
            #     model.save(MODEL_NAME)
        except Exception as e:
            print(str(e))
