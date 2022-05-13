import numpy as np 
import math
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1-s)

class network():
    def __init__(self, net_layers, learning_rate = 0.01,max_iterations=5000):
        #creating the layers from user input
        network_config = []
        i=0
        for i in range(len(net_layers)-2):
            network_config.append(weighted_layer(net_layers[i],net_layers[i+1]))
            network_config.append(activation_layer())
        i+=1
        network_config.append(weighted_layer(net_layers[i],net_layers[i+1]))

        self.net_conf = network_config
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def predict(self,x):
        y_pred = x 
        for layer in self.net_conf:
            y_pred = layer.forward_prop(y_pred) 
        return y_pred

    def train_function(self,samples,target):
        print("training the net...")
        i=0
        mse=100
        while mse>0.03 and i<self.max_iterations:
            i+=1
            mse = 0
            for sample_index in range(len(samples)):
                s = samples[sample_index]
                s = np.reshape(s,(len(s),1))
                #forward
                y_pred = self.predict(s)
                y = target[sample_index]

                #mean squared error
                mse += np.mean(np.power(y-y_pred,2))
                
                #backprop
                gradient = 2 * (y_pred-y) / np.size(y)
                for layer in reversed(self.net_conf):
                    gradient = layer.backward_prop(gradient,self.learning_rate)
            mse = mse / len(samples)
        print("traning finished after", i , "epochs")
        
    def make_predictions(self,X_test):
        y_pred = []
        y_pred_cont = []
        for sample in X_test:
            sample = np.reshape(sample,(len(sample),1))
            pred = self.predict(sample)
            y_pred_cont.append(round(float(pred),2))
            if pred < 0.5:
                pred = 0
            elif pred > 1.5:
                pred = 2
            else:
                pred = 1
            y_pred.append(int(pred))
        return np.array(y_pred),np.array(y_pred_cont)

class weighted_layer():
    def __init__(self, neurons_input, neurons_output, X=None, Y=None):
        self.X = X  #input vector
        self.Y = Y  #output vector
        self.W = np.random.randn(neurons_output,neurons_input) #weights matrix
        self.b = np.random.randn(neurons_output,1)  #biases vector

    def forward_prop(self,y_prev):
        self.X = y_prev
        return np.dot(self.W,self.X) + self.b

    def backward_prop(self,grad_succ,learning_rate):  #sposta il lr globale
        #backpropagation alg
        w_gradient = np.dot(grad_succ,self.X.T)
        grad_pred = np.dot(self.W.T,grad_succ)  #gradient to return to the previous layer
        self.W = self.W - w_gradient * learning_rate  #update the weights
        self.b = self.b - grad_succ * learning_rate #update the bias 
        return grad_pred


class activation_layer():
    #class to apply the activation function
    def __init__(self, act_function = sigmoid, act_function_der=sigmoid_der, X=None, Y=None):
        self.X = X  #input vector
        self.Y = Y  #output vector
        self.act_function = act_function 
        self.act_function_der = act_function_der #first derivative of act function
    
    def forward_prop(self, y_prev):
        self.X = y_prev
        return self.act_function(self.X)

    def backward_prop(self, grad_succ, learning_rate):
        return np.multiply(grad_succ, self.act_function_der(self.X))


#split the data
X,Y = data["data"], data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

#set the layers of the network
net_layers = [4,10,5,7,1]
neural_network = network(net_layers)

#train the network
neural_network.train_function(X_train,Y_train)

#test
Y_pred, pred_cont = neural_network.make_predictions(X_test)
print(">> ys predicted by the model (without rounding): ",pred_cont)
print(">> ys predicted by the model with rounding: ",Y_pred)
print(">> ys expected from test data: ",Y_test)
print(">> Accuracy: ",accuracy_score(Y_test, Y_pred)*100,"%")    
