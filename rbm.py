import numpy as np
import time

class RBM:
  
    def __init__(self, num_visible, num_hidden, learning_rate = 0.1, momentum = 0.9, gibs_sampling_len = 30):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gibs_sampling_len = gibs_sampling_len
        self.last_error = 0

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)
        #print(self.weights.mean())
        self.prevd = np.zeros((self.num_visible + 1, self.num_hidden + 1)) 


    def train(self, all_data, batch_size = 100, max_epochs = 1000):
        errors = []
        all_data_len = len(all_data)
        for epoch in range(max_epochs):      
            error = 0;
            st = time.time()
            adn = list(range(0, all_data_len, batch_size))
            for dn in adn:
                data = all_data[dn : dn + batch_size]
                # Insert bias units of 1 into the first column.
                data = np.insert(data, 0, 1, axis = 1)
                visible_probs = data

                for k in range(0, self.gibs_sampling_len + 1):
                    [hidden_states, hidden_probs]= self.calc_hidden(visible_probs)
                    
                    if k == self.gibs_sampling_len:
                        neg_associations = np.dot(visible_probs.T, hidden_probs)
                        continue

                    if k == 0:
                        pos_associations = np.dot(visible_probs.T, hidden_probs)
                      
                    visible_probs = self.calc_visible(hidden_states)
                    # visible_probs = self.calc_visible(hidden_probs)

                # Update weights.
                self.prevd = self.learning_rate * (pos_associations - neg_associations) / batch_size + self.momentum * self.prevd
                self.weights += self.prevd
                error += np.sum((data - visible_probs) ** 2)
            
            et = time.time()
            print("time %f" % (et - st))
            error /= all_data_len
            errors.append(error)
            self.last_error = error
            print("Epoch %s: error is %s" % (epoch, error))
            #print(self.weights.mean())
            #print(self.weights[1:self.weights.shape[0],1:self.weights.shape[1]].mean())

    def calc_hidden(self, visible_states):
        hidden_activations = np.dot(visible_states, self.weights)      
        hidden_probs = self._logistic(hidden_activations)
        hidden_probs[:,0] = 1 # Fix the bias unit.
        hidden_states = hidden_probs > np.random.rand(len(visible_states), self.num_hidden + 1)
        return [hidden_states, hidden_probs]

    def calc_visible(self, hidden_states):
        visible_activations = np.dot(hidden_states, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_probs[:,0] = 1 # Fix the bias unit.
        return visible_probs

    def save(self, fname):
        np.save(fname, self.weights)

    def load(self, fname):
        self.weights = np.load(fname)
                      
    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

