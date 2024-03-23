import numpy as np  # Import the numpy library for numerical operations
import argparse  # Import the argparse library for parsing command line arguments
import cv2  # Import the OpenCV library for image processing
from keras.layers import Conv2D  # Import the Conv2D layer from Keras for convolutional operations
import tensorflow as tf  # Import the TensorFlow library for deep learning
from keras.optimizers import SGD  # Import the SGD optimizer from Keras for stochastic gradient descent
from sklearn.datasets import fetch_openml  # Import the fetch_openml function from scikit-learn to download the MNIST dataset
from sklearn.model_selection import train_test_split  # Import the train_test_split function from scikit-learn to split the dataset into training and testing sets
from cnn.neural_network import CNN  # Import the CNN class from the neural_network module
# from sklearn.datasets import fetch_mldata  # Deprecated function to fetch MNIST dataset

# Parse the Arguments
ap = argparse.ArgumentParser()  # Create an ArgumentParser object to handle command line arguments
ap.add_argument("-s", "--save_model", type=int, default=-1)  # Add an argument to save the model
ap.add_argument("-l", "--load_model", type=int, default=-1)  # Add an argument to load a pre-trained model
ap.add_argument("-w", "--save_weights", type=str)  # Add an argument to save the weights of the model
args = vars(ap.parse_args())  # Parse the command line arguments and store them in a dictionary

# Read/Download MNIST Dataset
print('Loading MNIST Dataset...')
# dataset = fetch_mldata('MNIST Original')
dataset = fetch_openml('mnist_784')

# Read the MNIST data as array of 784 pixels and convert to 28x28 image matrix 
# Reshape the MNIST data to have dimensions (num_samples, 28, 28)
mnist_data = dataset.data.values.reshape((dataset.data.shape[0], 28, 28))

# Add a new axis to the data to represent the channel dimension
mnist_data = mnist_data[:, np.newaxis, :, :]

# Split the data into training and testing sets
from keras.utils import to_categorical  # Import the to_categorical function from Keras

train_img, test_img, train_labels, test_labels = train_test_split(mnist_data/255.0, dataset.target.astype("int"), test_size=0.1)

# Set the image dimensions
img_rows, img_columns = 28, 28

# Convert the labels to categorical format
total_classes = 10    #总类别数0-9
train_labels = to_categorical(train_labels, 10)    #测试数据和训练数据都从类别向量（整数）转化为二进制类别矩阵
test_labels = to_categorical(test_labels, 10)

# Defing and compile the SGD optimizer and CNN model
print('\n Compiling model...')
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
clf = CNN.build(width=28, height=28, depth=1, total_classes=10, Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Initially train and test the model; If weight saved already, load the weights using arguments.
b_size = 128		# Batch size
num_epoch = 20		# Number of epochs完整训练次数
verb = 1			# Verbose输出记录

# If weights saved and argument load_model; Load the pre-trained model.如果前面有把模型保存起来，那么就加载前面训练的模型
if args["load_model"] < 0:
	print('\nTraining the Model...')
	clf.fit(train_img, train_labels, batch_size=b_size, epochs=num_epoch,verbose=verb)
	
	# Evaluate accuracy and loss function of test data
	print('Evaluating Accuracy and Loss Function...')
	loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
	print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

	
# Save the pre-trained model.
if args["save_model"] > 0:
	print('Saving weights to file...')
	clf.save_weights(args["save_weights"], overwrite=True)

	
# Show the images using OpenCV and making random selections.
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
	# Predict the label of digit using CNN.
	probs = clf.predict(test_img[np.newaxis, num])
	prediction = probs.argmax(axis=1)

	# Resize the Image to 100x100 from 28x28 for better view.
	image = (test_img[num][0] * 255).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# Show and print the Actual Image and Predicted Label Value
	print('Predicted Label: {}, Actual Value: {}'.format(prediction[0],np.argmax(test_labels[num])))
#     cv2.imshow('Digits', image)
#     cv2.waitKey(0)

#---------------------- EOC ---------------------
