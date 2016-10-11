# Copyright (c) 2012-2016 Stuart Riffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import os

inputs = np.zeros( (len( positions ), 64 * 6), dtype = 'int8' )
outputs = np.zeros( len( positions ) )

for i, pos in enumerate( positions ):
    inputs[i] = pos[0]
    outputs[i] = pos[1]

print( 'Here we go' )

model = Sequential()

model.add( Dense( 512, input_dim = 64 * 6 ) )
model.add( Activation( 'sigmoid' ) )
model.add( Dropout( 0.5 ) )

model.add( Dense( 64 * 6 ) )
model.add( Activation( 'sigmoid' ) )

model.add( Dense( 1 ) )
model.add( Activation( 'linear' ) )


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, class_mode="binary")

history = model.fit(inputs, outputs, nb_epoch=10000, batch_size=1000, show_accuracy=True, verbose=1)

