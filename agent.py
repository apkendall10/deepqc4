import numpy as np, math, random, os, gameControls as gc, math
from joblib import load, dump
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from collections import deque

class agent:
    
    #initialize agent
    def __init__(self, fpath = None, lr = .001, debug = False, initial_explore_rate = .1, 
    recall_rate = 0, full_layer = True, advanced_memory = False, use_target = True, cnn = True):
        #initialize variables
        self.learning_rate = lr
        length = 1000
        self.memory = deque(maxlen=length)
        self.debug = debug
        self.explore_rate = initial_explore_rate
        self.error_baseline = .5
        self.recall_rate = recall_rate
        self.advanced_memory = advanced_memory
        self.use_target = use_target
        if self.advanced_memory:
            self.memory_score = deque(maxlen=length)
            self.memory_scale = []

        #load or build networks
        if fpath == None:
            self.NN = self.initialize(full_layer, cnn)
            self.target = self.initialize(full_layer, cnn)
            self.update_target(1)
        else:            
            fname = fpath + '/' + 'agent.joblib'
            try:
                self.NN = load(fname)
                self.target = load(fname)
            except:
                print('error reading file: ' + fname)
        self.NN.compile(loss="mean_absolute_error",
            optimizer=Adam(lr=self.learning_rate))

    def pick_move(self, board, player = 1):
        if random.random() > self.explore_rate:
            return self.forward_pass(board, player)
        else:
            return [random.choice(board.available_columns)]

    #feed forward pass to pick the move that maximized the expected outcome
    def forward_pass(self, board, player = 1):
        b = np.reshape(board.state * player,(1,6,7,1))
        moves = self.NN.predict(b)[0]
        index = np.argmax(moves[board.available_columns])
        pick = board.available_columns[index]
        #print(moves, index, pick, board.available_columns)
        game_details = [pick]
        return game_details

    #back propagation based on game outcome
    def learn(self, new_memory, player=1, use_memory = True, clip = False, use_target = True):
        #initialize memory and determine history
        errors = []
        if use_memory:
            for mem in new_memory:
                self.memory.append(mem)
                if self.advanced_memory:
                    self.memory_score.append(1)

            mem_count = min(32,len(self.memory))

            if self.advanced_memory:
                self.memory_scale = []
                cumulative_score = 0
                for score in self.memory_score:
                    cumulative_score += score
                    self.memory_scale.append(cumulative_score)
                self.memory_scale = np.array(self.memory_scale)
                #print(self.memory_scale)
                history = []
                for m in range(mem_count):
                    score = random.random() * cumulative_score
                    pick = np.argwhere(self.memory_scale > score)[0][0]
                    #print(score,pick)
                    history.append(pick)

            else:
                history = random.sample(self.memory, mem_count)

        else:
            history = new_memory
        
        n = len(history)

        if self.debug:
            print('Memory Size', len(self.memory))

        #use history for learning
        for mem in history:
            
            mems = self.memory[mem] if self.advanced_memory else mem

            flip = random.random() < .5
            b = mems[0]
            move = mems[1]
            if flip:
                b = b.flip()
                move = 6-move
            board = np.reshape(b.state * player, (1,6,7,1))
            reward = mems[2] * player
            done = False
            if self.use_target:
                nb = mems[3]
                if flip:
                    nb = nb.flip()
                next_b = np.reshape(nb.state * player , (1,6,7,1))
                available_columns = nb.available_columns
                done = mems[4]
            
            #calculate error
            if self.use_target:
                target = self.target.predict(board)[0]
            else:
                target = self.NN.predict(board)[0]
            
            previous_val = target[move]

            if not self.use_target or done:
                target[move] = reward
            else:
                future = self.target.predict(next_b)[0]
                q_future = max(future[available_columns])
                target[move] = q_future * .95
                
                if self.debug:
                    print(future,available_columns)
            
            if self.debug:
                b.show_board()
                if self.use_target:
                    nb.show_board()
                pred = self.NN.predict(board)[0]
                print('before',move,pred[move],target[move],pred)

            error = abs(previous_val - target[move])
            errors.append(error)

            if self.advanced_memory:
                self.memory_score[mem] = math.sqrt(error)

            if error * self.recall_rate > random.random():
                self.memory.append(mems)
                #print('Memory Added', error)

            self.NN.fit(board, np.reshape(target,(1,7)), epochs = 1, batch_size = 1, verbose = False)
            
            if self.debug:
                pred = self.NN.predict(board)[0]
                print('after',pred[move],pred)

        errors = np.array(errors)
        avg_error = errors.mean()
        return avg_error

    #loop through memory for a while
    def replay(self, rounds = 1000, display = False, use_target = True):
        errors = []
        for i in range(rounds):
            errors.append(self.learn([], use_target=use_target))
        if display:
            errors = np.array(errors)
            return errors.mean()

    #saves agent weights to file
    def save_agent(self, fpath):
        fname = fpath + '/agent.joblib'
        try:
            dump(self.NN,fname)
        except:
            print('error saving file to ' + fname)

    #helper method to print weights for troubleshooting
    def print_weights(self):
            output = ''
            for layer in range(len(self.weights)):
                w = self.weights[layer]
                output += 'layer ' + str(layer) + '\n'
                for i in range(len(w)):
                    row = w[i,:]
                    for val in row:
                        output += ',' + str(round(val,4))
                    output += '\n'
            print(output)

    def random_train(self,rounds = 1000, use_memory = False, use_target = True):
        for i in range(rounds):
            game_data = gc.play_game(1, self)
            outcome = game_data[0]
            self.learn(game_data[1], 1, use_memory, use_target = use_target)
    
    def evaluate(self, rounds = 1000):
        wins = 0
        losses = 0
        for i in range(rounds):
            game_data = gc.play_game(1,self)
            outcome = game_data[0]
            if outcome == 1:
                wins += 1
            elif outcome == -1:
                losses += 1
        return (wins, losses, rounds - wins - losses)

    def update_target(self, tao):
        predict_weights = self.NN.weights
        target_weights = self.target.weights
        for i in range(len(target_weights)):
            target_weights[i] = predict_weights[i] * tao + target_weights[i] * (1-tao)
        self.target.set_weights(target_weights)
    #initialize a network of weights
    def initialize(self, full_layer = True, cnn = True):
        model = Sequential()
        if cnn:
            n_filters = 32 if full_layer else 16
            first_layer_stride = (1,1) if full_layer else (2,2)
            model.add(Conv2D(filters= n_filters, kernel_size = (2,2), strides = first_layer_stride, input_shape=(6,7,1), activation='relu'))
            model.add(Conv2D(filters= n_filters * 2, kernel_size = (2,2), strides = (1,1), activation='relu'))
            if full_layer:
                model.add(Conv2D(filters= n_filters * 4, kernel_size = (2,2), strides = (1,1), activation='relu'))
            
        else:
            model.add(Conv2D(filters = 1, kernel_size = (1,1), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=7))
        return model
