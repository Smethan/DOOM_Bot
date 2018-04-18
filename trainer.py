from models import inception_v3 as googlenet
from random import shuffle
import numpy as np
import time
import tensorflow as tf
import logging
import tflearn
import gc
tf.logging.set_verbosity(0)
tf.reset_default_graph()
with tf.Graph().as_default():
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
    tf.logging.set_verbosity(tf.logging.DEBUG)
    if GARBAGE_LOGGING:
        file = open('gc_log.txt', 'w')

    model = googlenet(WIDTH, HEIGHT, 1, LR, output=14, model_name=MODEL_NAME)

    # if LOAD_MODEL:
    #     model.load(PREVIOUS_MODEL)
    #     print('loaded previous model!')


    for e in range(EPOCHS):
        data_order = [i for i in range(1,FILE_END+1)]
        shuffle(data_order)
        for count,i in enumerate(data_order):

            try:

                file_name = 'training_data-{}.npy'.format(i)
                train_data = np.load(file_name)
                print('Loading training_data-{}.npy'.format(i))
                train = train_data[:-50]
                test = train_data[-50:]

                X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,2)
                Y = np.array([i[1] for i in train])

                test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,2)
                test_Y = np.array([i[1] for i in test])

                model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_X}, {'targets': test_Y}), snapshot_step=2500, show_metric=True,run_id=MODEL_NAME)

                gc.collect()
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

                if count%10 == 0:
                    print('SAVING MODEL!')
                    model.save(MODEL_NAME)
            except Exception as e:
                print(str(e))