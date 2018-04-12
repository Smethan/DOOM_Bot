from models import inception_v3 as googlenet
from random import shuffle
import numpy as np
import time

FILE_END = 39
WIDTH = 4
HEIGHT = 25
LR = 1e-3
EPOCHS = 30
MODEL_NAME='DOOM_Beta_Bot'
PREVIOUS_MODEL ='DOOM_Beta_Bot'
LOAD_MODEL = True

model = googlenet(WIDTH, HEIGHT, 3, LR, output=14, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREVIOUS_MODEL)
    print('loaded previous model!')


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
            Y = [i[1] for i in train]

            test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,2)
            test_Y = [i[1] for i in test]

            model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_X}, {'targets': test_Y}), snapshot_step=2500, show_metric=True,run_id=MODEL_NAME)

            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

        except Exception as e:
            print(str(e))