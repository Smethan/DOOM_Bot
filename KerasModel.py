from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Convolution2D, MaxPooling2D, AveragePooling1D



def KMod(width, height, output=14):
    model = Sequential()
    model.add(Convolution2D(6,1,1,activation='relu',input_shape=(1,width,height)))
    model.add(Convolution2D(6,1,1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
