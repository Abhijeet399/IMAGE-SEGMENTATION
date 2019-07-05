import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

voc_dir = "D:/VOC2012"

txt_fname = "D:/VOC2012/ImageSets/Segmentation"
with open(txt_fname, 'r') as f:
        images = f.read().split()

print("COPYING IMAGES AND LABELS")

labels = []
features = []
for i in range(len(images)):
    labels.append(cv2.resize(cv2.imread("D:/VOC2012/SegmentationClass/" + images[i] + ".png"), (512, 512)))
    features.append(cv2.resize(cv2.imread("D:/VOC2012/JPEGImages/" + images[i] + ".jpg"), (512, 512)))

print("LABELS IMAGES")

plt.imshow(features[53])
plt.imshow(labels[55])

def create_placeholders(n_H0, n_W0, n_C0, n_H1,n_W1,n_C1):

    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_H1,n_W1,n_C1])    
    return X, Y

def initialize_parameters():
 
    tf.set_random_seed(1)                              # so that your "random" numbers match ours  
    W1 = tf.get_variable("W1", [6, 6, 3, 8], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3, 3, 8, 42], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [2, 2, 42, 42], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [2, 2, 42, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable("W5", [6, 6, 8, 3], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W6 = tf.get_variable("W6", [6, 6, 8, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5}
    return parameters

def forward_propagation(X, parameters):
     
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']
            # CONV2D: 
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'VALID')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL:
    P1 = tf.nn.max_pool(A1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D:
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,2,2,1], padding = 'VALID')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL:
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # FLATTEN
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,2,2,1], padding = 'SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.conv2d_transpose(A3,W4,[100,100,8],[1,3,3,1],padding='SAME')
    A4 = tf.nn.relu(Z4)
    Z6 = tf.nn.conv2d_transpose(A4,W6,[258,258,16],[1,3,3,1],padding='SAME')
    A6 = tf.nn.relu(Z6)
    Z5 = tf.nn.conv2d_transpose(A6,W5,[512,512,3],[1,3,3,1],padding='SAME')
    A5 = tf.nn.relu(Z5)
    return A5

def compute_cost(A5, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = A5, labels = Y))    
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
      
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3      
    m=2913                                    # to keep results consistent (numpy seed)
    (n_H0, n_W0, n_C0) = X_train.shape             
    (n_H1,n_W1,n_C1) = Y_train.shape                           
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_H1,n_W1,n_C1)
    # Initialize parameters
    parameters = initialize_parameters()     
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A5 = forward_propagation(X, parameters)    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(A5, Y)    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    seed = 0
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            # minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                minibatch_X = X_train[(seed*minibatch_size):((seed*minibatch_size)+minibatch_size)]
                minibatch_Y = Y_train[(seed*minibatch_size):((seed*minibatch_size)+minibatch_size)]
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
            
            seed = seed + 1    

            # Print the cost every epoch
            #if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            # if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(A5, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        # train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        # test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        # print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters