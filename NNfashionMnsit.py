'''
    
    @author: Jordan Walker
    
    @Code Sources:  (Raschka and Mirajalili, n.d.)
                    (Hong, 2018)
                    
    @Data Sources: Fashion MNIST - Marta, Heriot Watt University
    All sources are harvarded referenced in the report under ther reference section
    Data Files: Marta, Heriot Watt University
    Model:
    
    '''


import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit

#################
### Load Date ###
#################

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
                               
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)


    return images, labels

#################
## Check Loaded #
#################

x_train, y_train = load_mnist('', kind='train')
print('x train: Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
x_test, y_test = load_mnist('', kind='t10k')
print('x test: Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

#################
## Normalize  ###
#################

x_train = x_train / 255;
x_test = x_test / 255;

#################
## Neural Net ###
#################

class NeuralNetMLP(object):
    #Specifies the initial values of the neural network
    def __init__(NN,x_train,y_train,x_test, y_test, n_hidden=1,l2=0.1, epochs=1, LearnR=0.001,shuffle=True, minibatch_size=1, seed=None):
        
        #Initalize Neural Network Values
        NN.random = np.random.RandomState(seed)
        NN.n_hidden = n_hidden
        NN.l2 = l2
        NN.epochs = epochs
        NN.LearnR = LearnR
        NN.shuffle = shuffle
        NN.minibatch_size = minibatch_size
    
    def onehot(NN, y, n_classes):
        #Produces a vector in accordance with the class
        
        onehot = np.zeros((n_classes, y.shape[0]))
        for idOx, val in enumerate(y):
            onehot[val, idOx] = 1.
        return onehot.T
    
    def sigmoid(NN, z):
        #Activation Function used in this model
        s = 1. / (1. + np.exp(-z));
        return s;
    
    def fPropogate(NN, x):
        #Forward propogation of the Network
        
        z_h = np.dot(x, NN.w_h) + NN.b_h #Hidden Layer Input
        a_h = NN.sigmoid(z_h) #Hidden Layer Activation
        z_out = np.dot(a_h, NN.w_out) + NN.b_out #Input of Output Layer
        a_out = NN.sigmoid(z_out) #Output Layer Activation
        return z_h, a_h, z_out, a_out
    
    def compute_cost(NN, y_enc, output):
        #Where I calcualte the different regularization methods
        L1R = (NN.l2 *(np.sum(NN.w_h) + np.sum(NN.w_out)))
        #L2R = (NN.l2 *(np.sum(NN.w_h ** 2.) + np.sum(NN.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        #cost = np.sum(term1 - term2) + L2R
        cost = np.sum(term1 - term2) + L1R
        return cost

    def predict(NN, x):
        #predict the output class labels
        z_h, a_h, z_out, a_out = NN.fPropogate(x)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(NN, x_train, y_train, x_valid, y_valid):
        #learn weights from the training data data
        n_output = np.unique(y_train).shape[0]
        n_features = x_train.shape[1]
        
        #weights for hidden layer
        NN.b_h = np.zeros(NN.n_hidden)
        NN.w_h = NN.random.normal(loc=0.0, scale=0.1,size=(n_features, NN.n_hidden))
  
        #weights from hidden to output layer
        NN.b_out = np.zeros(n_output)
        NN.w_out = NN.random.normal(loc=0.0, scale=0.1,size=(NN.n_hidden, n_output))
        epoch_strlen = len(str(NN.epochs))
        NN.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = NN.onehot(y_train, n_output)
  
        for i in range(NN.epochs):

            indices = np.arange(x_train.shape[0])
                                          
            if NN.shuffle:
                NN.random.shuffle(indices)
                    
            for start_idx in range(0, indices.shape[0] - NN.minibatch_size, NN.minibatch_size):
                batch_idx = indices[start_idx:start_idx + NN.minibatch_size]

                z_h, a_h, z_out, a_out = NN.fPropogate(x_train[batch_idx])
                
                
                #Begin Back Propogation
                
                sigma_out = a_out - y_train_enc[batch_idx]

                sigmoid_derivative_h = a_h * (1. - a_h)

                sigma_h = (np.dot(sigma_out, NN.w_out.T) * sigmoid_derivative_h)

                grad_w_h = np.dot(x_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)
                 
                 #regularization and weight updates for the Neural Network
                delta_w_h = (grad_w_h + NN.l2*NN.w_h)
                delta_b_h = grad_b_h
                NN.w_h -= NN.LearnR * delta_w_h
                NN.b_h -= NN.LearnR * delta_b_h
                                                                             
                delta_w_out = (grad_w_out + NN.l2*NN.w_out)
                delta_b_out = grad_b_out
                NN.w_out -= NN.LearnR * delta_w_out
                NN.b_out -= NN.LearnR * delta_b_out
    
            #Begin Evaluation
            z_h, a_h, z_out, a_out = NN.fPropogate(x_train)
                                                                                         
            cost = NN.compute_cost(y_enc=y_train_enc,output=a_out)

            y_train_pred = NN.predict(x_train)
            y_valid_pred = NN.predict(x_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /x_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /x_valid.shape[0])
                                                                                                                                    
            print('\r Epoch: %0*d/%d  Cost: %.2f '' Train/Valid Acc: %.2f%%/%.2f%% ' % (epoch_strlen, i+1, NN.epochs, cost, train_acc*100, valid_acc*100))
           
            NN.eval_['cost'].append(cost)
            NN.eval_['train_acc'].append(train_acc)
            NN.eval_['valid_acc'].append(valid_acc)

        return NN

#################
#### Testing ####
#################
nn = NeuralNetMLP(x_train, y_train, x_test, y_test, n_hidden=49,l2=0.0,epochs=200,LearnR=0.0001,minibatch_size=600,shuffle=True,seed=1);
nn.fit(x_train,y_train,x_test,y_test)

y_test_pred = nn.predict(x_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / x_test.shape[0])
print('Test accuracy: %.2f%%' % (acc * 100))

"""
plt.plot(range(nn.epochs), nn.eval_['cost'])


plt.plot(range(nn.epochs), nn.eval_['train_acc'])
plt.plot(range(nn.epochs), nn.eval_['valid_acc'])



plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()



hiddenNeurons = [784, 392, 196, 98, 49];
learningRates = [0.01, 0.001, 0.0001];
RegressionRates = [0.0, 0.1, 0.01]
miniBatch = [250,550,750,100]

nn = {};

nn = NeuralNetMLP(x_train, y_train, x_test, y_test, n_hidden=49,l2=0.01,epochs=200,LearnR=0.001,minibatch_size=250,shuffle=True,seed=1);
plt.plot(range(nn.epochs), nn.eval_['cost'], label='training with HN ' + str(i))
print('Test accuracy: %.2f%%' % (acc * 100))
plt.ylabel('cost')
plt.xlabel('Epochs')
legend = plt.legend(loc='best', shadow=True);
plt.show();

for i in miniBatch:
    print ("Mini Batch Size is: " + str(i));
    start = timeit.default_timer()
    nn = NeuralNetMLP(x_train, y_train, x_test, y_test, n_hidden=49,l2=0.01,epochs=200,LearnR=0.001,minibatch_size=i,shuffle=True,seed=1);
    nn.fit(x_train,y_train,x_test,y_test)
    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
    
    plt.plot(range(nn.epochs), nn.eval_['cost'], label='training with HN ' + str(i))
    
    plt.plot(range(nn.epochs), nn.eval_['train_acc'],label='training with HN ' + str(i))
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'],label='validation with HN ' + str(i), linestyle='--')

    y_test_pred = nn.predict(x_test)
    acc = (np.sum(y_test == y_test_pred).astype(np.float) / x_test.shape[0])
    print('Test accuracy: %.2f%%' % (acc * 100))

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
legend = plt.legend(loc='best', shadow=True);
plt.show();
"""








