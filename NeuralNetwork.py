import pandas as pd
import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # Weight initilaization should be a function of the input layer size (1/sqrt(n))
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # Use sigmoid as hidden layer activation function
        self.activation_function = lambda x: 1/(1+np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''

        n_records = features.shape[0]
        # Initial delta weights as zeros
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            
            # Forward pass
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            # Backpropagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
                            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        hidden_input = np.dot(X, self.weights_input_to_hidden) # h = W*x
        hidden_output = self.activation_function(hidden_input) # f(h)
        output = np.dot(hidden_output, self.weights_hidden_to_output) # Linear Output activation function f(x) = x
        return output, hidden_output

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # Calculate error
        error = y - final_outputs
        # No Non-linearity for error term (delta2), gradient == 1
        output_error_term = error

        # Hidden layer error contribution: error = (delta2 * weights)
        hidden_error = np.dot(output_error_term, delta_weights_h_o)

        # Backpropagate hiden error to input layer: hidden_error_term (or delta1) = (error * gradient)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

        # Update weigths
        delta_weights_i_h += np.dot(hidden_error_term, X)
        delta_weights_h_o += np.dot(output_error_term, hidden_outputs)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += delta_weights_h_o * self.lr / n_records
        self.weights_input_to_hidden += delta_weights_i_h * self.lr / n_records
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        pass

