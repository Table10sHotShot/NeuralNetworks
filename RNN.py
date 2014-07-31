
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def logistic_fn(z):
    return 1/(1+np.exp(-z))

def dlogistic_fn(z):
    return logistic_fn(z)*(1-logistic_fn(z))

class RNN(object):

    def __init__(self, num_hidden_neurons):

        self._num_hidden_neurons = num_hidden_neurons
        
#        initialize weights
        self._ff_weights = []
        self._ff_weights.append(np.random.rand(3, num_hidden_neurons))
        self._ff_weights.append(np.random.rand(num_hidden_neurons+1, 1))

        self._time_weights = [np.random.rand(num_hidden_neurons, num_hidden_neurons), np.random.rand(1,1)]

        self._x_hidden_init = np.append([1], np.random.rand(num_hidden_neurons))
        self._x_output_init = np.array([.5])

#    reset training example dependent variables
    def reset(self):

        self._xs_input = []
        self._xs_hidden = [self._x_hidden_init]
        self._xs_output = []#self._x_output_init]
        self._zs_hidden = []
        self._zs_output = []

        self._ff_weight_gradient = [np.zeros((3,self._num_hidden_neurons)), np.zeros((self._num_hidden_neurons+1,1))]
        self._time_weight_gradient = [np.zeros((self._num_hidden_neurons, self._num_hidden_neurons)), np.zeros((1,1))]

    def feed_forward(self, input_sequence1, input_sequence2):

        for int1, int2 in zip(input_sequence1, input_sequence2):
#            create input array
            x_input = np.array([1,int1, int2])
            self._xs_input.append(x_input)
            
#            calculate signal going into hidden layer - includes input from previous timestep hidden layer
            z_hidden = np.dot(x_input, self._ff_weights[0]) + np.dot(self._xs_hidden[-1][1:], self._time_weights[0])
            self._zs_hidden.append(z_hidden)
            
#            calculate hidden layer output
            x_hidden = np.append([1], logistic_fn(z_hidden))
            self._xs_hidden.append(x_hidden)

#            calculate signal going into ouput layer - includes input from previous timestep output layer
            z_output = np.dot(x_hidden, self._ff_weights[1])# + np.dot(self._xs_output[-1], self._time_weights[1])
            self._zs_output.append(z_output)

#            calculate output layer output
            x_output = logistic_fn(z_output)
            self._xs_output.append(x_output)

    def back_propagate(self, targets, learning_rate):

        enumerated_targets = list(enumerate(targets))
        
        for timestep, target in reversed(enumerated_targets):

#            calculate dedz for output layer
            output = self._xs_output[timestep]
            z_output = self._zs_output[timestep]
            dedz_output = (output - target) * dlogistic_fn(z_output)

#            calculate dedz for output layer assuming z is actual output
#            output = self._zs_output[timestep]
#            dedz_output = (output - target)

#            calculate dedx for hidden layer
            dedx_hidden = np.dot(self._ff_weights[1], dedz_output)
            
#            calculate dedz for hidden layer
            z_hidden = self._zs_hidden[timestep]
            dedz_hidden = dedx_hidden[1:] * dlogistic_fn(z_hidden)

#            calculate dedws
            dedw_ff_output = np.outer(self._xs_hidden[timestep+1], dedz_output)
            dedw_ff_hidden = np.outer(self._xs_input[timestep], dedz_hidden)
#            dedw_time_output = np.outer(self._xs_output[timestep], dedz_output)
            dedw_time_hidden = np.outer(self._xs_hidden[timestep][1:], dedz_hidden)
            
#            update weight gradients
            self._ff_weight_gradient[1] += dedw_ff_output
            self._ff_weight_gradient[0] += dedw_ff_hidden
#            self._time_weight_gradient[1] += dedw_time_output
            self._time_weight_gradient[0] += dedw_time_hidden
        
#        update initial output activity vector
        dedx_output_init = np.dot(self._time_weights[1], dedz_output)
        self._x_output_init -= learning_rate * dedx_output_init

#        update initial hidden activity vector
        dedx_hidden_init = np.dot(self._time_weights[0], dedz_hidden)
        dedx_hidden_init = np.append([0], dedx_hidden_init)
        self._x_hidden_init -= learning_rate * dedx_hidden_init

        eff_learning_rate = learning_rate / len(enumerated_targets)

#        update weights
        self._ff_weights[0] -= eff_learning_rate * self._ff_weight_gradient[0]
        self._ff_weights[1] -= eff_learning_rate * self._ff_weight_gradient[1]
        self._time_weights[0] -= eff_learning_rate * self._time_weight_gradient[0]
#        self._time_weights[1] -= eff_learning_rate * self._time_weight_gradient[1]

    def train_sequence(self, input_sequence1, input_sequence2, target_sequence, learning_rate):

        self.feed_forward(input_sequence1, input_sequence2)

        self.back_propagate(target_sequence, learning_rate)

if __name__ == "__main__":

    logging.basicConfig(format='{levelname}: {message}', style='{', level=logging.INFO)

    pkl_filename = 'rnn_4hu.pkl'

    if os.path.isfile(pkl_filename):
        with open(pkl_filename, 'rb') as f: rnn = pickle.load(f)
        logger.info("loaded rnn")
        logger.debug('ff weights', str(rnn._ff_weights))
        logger.debug('time weights', str(rnn._time_weights))
    else: rnn = RNN(4)

    num_training_exs = 2e4
    learning_rate = .04
    
    i=0
    for num1, num2 in np.random.randint(0,63, (num_training_exs,2)):
        
        i+=1
        perc = i / num_training_exs
        if not (perc * 100 % 10): logger.debug(perc)

        num_target = num1+num2
        
        str1 = bin(num1)[2:]
        str2 = bin(num2)[2:]
        str_target = bin(num_target)[2:]

        str_len = max(len(str1), len(str2)) + 1

        str1 = str1.rjust(str_len, '0')
        str2 = str2.rjust(str_len, '0')
        str_target = str_target.rjust(str_len, '0')

        input_sequence1 = reversed(list(map(int, str1)))
        input_sequence2 = reversed(list(map(int, str2)))
        target_sequence = reversed(list(map(int, str_target)))

        rnn.reset()
        rnn.train_sequence(input_sequence1, input_sequence2, target_sequence, learning_rate)

        if num_training_exs - i < 10:
            logger.debug(str1 + ' ' + str2 + ' ' + str_target)
            predicted = ''.join(list( map(lambda x: str(int(round(x))), np.ravel(rnn._xs_output)[::-1]) ))
            logger.debug(predicted)
            logger.info(predicted == str_target)

    with open(pkl_filename, 'wb') as f: pickle.dump(rnn, f)
