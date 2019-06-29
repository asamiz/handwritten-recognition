""" 
         Application to classify a handwritten digits using neural networks.By using libraries like keras and tensorflow
         and MNIST dataset to train and test our NN.
         Plotting and Sample of the code copyrights to https://dzubo.github.io/keras/2017/04/21/tutorial-on-using-keras-for-digits-recognition.html
         THANK YOU FOR SHARING THAT! .. FROM ZAGAZIG UNIVERSITY(CSE 4th YEAR 2018!) - SHARKIA GOV. - EGYPT 
"""

                #------------------------------------------------#
                                # About the Application
               #------------------------------------------------#
""" 
        Creating the NN model and determining the number of hidden layer neurons and
        activation functions to be used like (Relu or Sigmoid) in both hidden layer neurons and
        Output neurons.
        Also with determination of number of Epcohs and Batch size (How many input in one iteration).
        The cost function used is "crossentropy function" and the metric used to measure performance is'Accuracy'
        The number of output neurons and the inputs are fixed as the application dealing with 10 classes 
        the numbers from 0 to 9 and input images 28*28 which means 784 value in each row of pixels.
        We can check the performance from the vaule of accuracy but it WILL NOT be accurate so visualizing
        the results will be useful in this case!
"""

#------------------------------------------------#
               # Importing the libraries
#------------------------------------------------#
get_ipython().run_line_magic('matplotlib', 'auto')
import pandas as pd      # To import datasets
import matplotlib.pyplot as plt 
# -----------------------------------------------#
from keras.models import Sequential    # To make the NN model with sequential layers (Input - ... Hidden ... - output)
from keras.layers import Dense, Activation  # To cnostruct the neural (layers, No. of neurons and activation functions)
#------------------------------------------------#

#------------------------------------------------#
               # Importing the Data  
#------------------------------------------------#
# Importing training dataset.
train_df = pd.read_csv("train.csv")
# Importing testing dataset.
test_df = pd.read_csv("test.csv")
#------------------------------------------------#

#------------------------------------------------#
               # Data_preprocessing1
#------------------------------------------------#
# Dividin the data into images and the number(label) corresponding to the image
# First the label of the images or the number coreesponding to the pixels of the image
train_labels = train_df.label
# The image (pixels that constructing the image 28*28)
train_images = train_df.iloc[:,1:]
# Test image or pixels to be predicted without labels
test_images = test_df
#------------------------------------------------#

#------------------------------------------------#
             # Data_preprocessing2
#------------------------------------------------#             
# Normalizing the data.
train_images = (train_images/train_images.max()).fillna(0) 
test_images = (test_images/test_images.max()).fillna(0) 
# Applying one_hot_encoder to the labels (numbers)
train_labels = pd.get_dummies(train_labels) 
#------------------------------------------------#
              
#------------------------------------------------#
              # Viualizing_Data(Training)
#------------------------------------------------#              
# Visualizing sample of trainig data.
plt.figure(figsize=(12,8))
for i in range(0,9):
    plt.subplot(250 + (i+1))
    img = train_images.ix[i,:].values.reshape(28, 28)
    plt.imshow(img, cmap='Greys')
#------------------------------------------------#
             

#------------------------------------------------#
              # Creating the NN model
              # Training the model
              # Plotting Loss Vs Epochs (Training & Testing)
              # Plotting Accuracy Vs Epochs (Training & Testing)
#------------------------------------------------#
def Neural_Network(Hidden_Neurons,Activation_Hidden,Activation_Output,Number_Epoch,Batch_Size):
    
    # Initializing and Creating our NN
    model = Sequential([Dense(Hidden_Neurons, input_dim=784),Activation(Activation_Hidden),Dense(10),Activation(Activation_Output)])
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    #--------------------------------------------#
    
    # Training the model
    history=model.fit(train_images.values, train_labels.values, validation_split = 0.05, nb_epoch=Number_Epoch, batch_size=Batch_Size)
    
    #--------------------------------------------#
    
    # Collecting the error and accuracy to be able to use it in the visualization step
    hist_df = pd.DataFrame(history.history)
    
    #--------------------------------------------#
    
    # Creating & Preparing the figure.
    fig = plt.figure(figsize=(20,14))
    plt.style.use('bmh')
    params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)
    
    #--------------------------------------------#
    
    # Plotting Loss Vs Epoch
    plt.subplot(121)
    plt.plot(hist_df.loss, label='Training loss', **params_dict)
    plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
    plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plotting accuracy Vs Epoch
    plt.subplot(122)
    plt.plot(hist_df.acc, label='Training accuracy', **params_dict)
    plt.plot(hist_df.val_acc, label='Validation accuracy', **params_dict)
    plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return model
    

# Calling NN function to make our neural .. YaY!
NN = Neural_Network(32,'relu','softmax',30,64) 

#------------------------------------------------#
                # Testing new values
#------------------------------------------------#                
def Predict_Fun ():
    # Testing the new values
    pred_classes = NN.predict_classes(test_images.values)
    return pred_classes
    #--------------------------------------------#
 
# Let's predict the new values!
pred_classes = Predict_Fun()    

  
#------------------------------------------------#    
                # Viualizing the result.
#------------------------------------------------#    
def Visualize_Test(result):

    # Visualizing the corresponding value according to result value entered (should be 0 - 28000)
    plt.figure(figsize=(10,4))
    img = test_images.ix[result,:].values.reshape(28, 28)
    plt.imshow(img, cmap='Greys')
    plt.show()
    print('should be = ', pred_classes[result] )
    plt.title(pred_classes[result])
        




# Calling Visualize_Test to visualize one of results
Visualize_Test(6541)





