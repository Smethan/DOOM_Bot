import urllib.request
import json
import os
from models import inception_v3 as googlenet
import numpy as np
import random

starting_value = 1
file_name = 'training_data-{}.npy'.format(starting_value)

#Inputs = {'UP': 72, 'DOWN': 80, 'LEFT': 75, 'RIGHT': 77, 'SHOOT': 29, 'INTERACT': 57, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9}
player_contents = json.loads(urllib.request.urlopen("http://localhost:8080/api/player").read().decode('utf-8'))
world_info = json.loads(urllib.request.urlopen("http://localhost:8080/api/world").read().decode('utf-8'))
world_contents = json.loads(urllib.request.urlopen("http://localhost:8080/api/world/objects?distance=200").read().decode('utf-8'))
Score = 0
WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30
model = googlenet(WIDTH, HEIGHT, 3, LR, output=9)
while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)

        break


def GetData():
    player_contents = json.loads(urllib.request.urlopen("http://localhost:8080/api/player").read().decode('utf-8'))
    # world_contents = json.loads(urllib.request.urlopen("http://localhost:8080/api/world/objects?distance=200").read().decode('utf-8'))
    important_data = [player_contents['kills'], player_contents['health'], player_contents['armor'], player_contents['angle']]
    return important_data
# example contents
# {'id': 0, 'position': {'x': 1056, 'y': -3616, 'z': 0}, 'angle': 90, 'height': 56, 'health': 100, 'typeId': -1, 'type': 'Player', 'flags': {'MF_SOLID': True, 'MF_SHOOTABLE': True, 'MF_DROPOFF': True, 'MF_PICKUP': True, 'MF_NOTDMATCH': True}, 'armor': 0, 'kills': 0, 'items': 0, 'secrets': 0, 'weapon': 1, 'keyCards': {'blue': False, 'red': False, 'yellow': False}, 'cheatFlags': {}}
def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    while True:
        keys = [random.randint(0,1) for x in range(14)]
        GameData = GetData()
        training_data.append([GameData,keys])
        if len(training_data) % 100 == 0:
            print(len(training_data))

            if len(training_data) == 500:
                np.save(file_name, training_data)
                print('SAVED')
                training_data = []
                starting_value += 1
                file_name = 'training_data-{}.npy'.format(starting_value)




main(file_name, starting_value)


