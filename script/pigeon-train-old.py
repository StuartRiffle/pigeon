# Copyright (c) 2012-2016 Stuart Riffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import chess.pgn
import os


def load_pgn_games( filename ):
    result = []
    with open( filename ) as pgn:
        while True:
            game = chess.pgn.read_game( pgn )
            if game is not None:
                result.append( game )
                if( len( result ) > 20 ):
                    break
            else:
                break
    return result            
    
    
def encode_board_position( str ):

    grid = np.zeros( 64 * 6, dtype = 'int8' )
    
    for i, c in enumerate( str ):
        if   c == 'P': grid[(64 * 0) + i] =  1
        elif c == 'p': grid[(64 * 0) + i] = -1
        elif c == 'N': grid[(64 * 1) + i] =  1
        elif c == 'n': grid[(64 * 1) + i] = -1
        elif c == 'B': grid[(64 * 2) + i] =  1
        elif c == 'b': grid[(64 * 2) + i] = -1
        elif c == 'R': grid[(64 * 3) + i] =  1
        elif c == 'r': grid[(64 * 3) + i] = -1
        elif c == 'Q': grid[(64 * 4) + i] =  1
        elif c == 'q': grid[(64 * 4) + i] = -1
        elif c == 'K': grid[(64 * 5) + i] =  1
        elif c == 'k': grid[(64 * 5) + i] = -1
            
    return grid

    
def extract_game_positions( game ):

    game_result = game.headers["Result"]
    
    if game_result == '1-0':
        expected = 1.0
    elif game_result == '0-1':
        expected = 0.0
    else:
        expected = 0.5
    
    result = []    
    
    node = game
    while not node.is_end():
        next_node = node.variation( 0 )
        board_state = str( node.board() ).replace( ' ', '' ).replace( '\n', '' )
        grid = encode_board_position( board_state )
        result.append( (grid, expected) )
        
        #print( board_state )
        expected = 1.0 - expected
        node = next_node
        
    print( len( result ) )
    return result
    
    

games = load_pgn_games( os.path.normpath( 'd:/chess/pgn/capablanca.pgn' ) )

positions = []
for game in games:
    positions = positions + extract_game_positions( game )
    
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

