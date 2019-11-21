import io
import pickle 
import copy
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import pandas_ml as pml

''' Function to divide the data into corresponding batches'''
def getBatches(image_names, num_images, num_batches):
    for i in range(0, len(image_names), num_images):
        first_indices.append(i)
    first_indices.append(len(image_names))
    for i in range(0,len(image_names)):
        if i < num_batches:
            index_pairs.append([])
            index_pairs[i].append(first_indices[i])
            index_pairs[i].append(first_indices[i+1]-1)
        elif i == num_batches:
            break
    for i in range(0,len(index_pairs)):
        batch_keys = []
        for j in range(index_pairs[i][0], index_pairs[i][1]+1):
            batch_keys.append(image_names[j])
        image_keys.append(batch_keys)
    return image_keys


''' Function to get the labels corresponding to each image in a batch'''
def getLabels(train_labels, image_names, image_keys, num_images):
	for i in range(len(image_names)):
		image_IDs.append(train_labels.iloc[i,0])
		image_labels.append(train_labels.iloc[i,1])
	Labels = dict(zip(image_IDs,image_labels))
	total_labels = []
	for i in range(0, num_batches):
		batch_labels = []
		for j in range(0, num_images):
			batch_labels.append(Labels[image_keys[i][j]])
			batch_labels_nparray = np.asarray(batch_labels)
		batch_labels_reshaped = np.reshape(batch_labels_nparray, (num_images,1))
		total_labels.append(batch_labels_reshaped)
	return np.asarray(total_labels)


''' Getting the features of the corresponding image from the train_features dictionary'''
def getFeatures(train_features, num_batches, image_keys, num_images):
    total_features = []
    for i in range(0, num_batches):
        batch_features = []
        for j in range(0, num_images):
            data_point = train_features[image_keys[i][j]]
            batch_features.append(data_point)
        total_features.append(np.asarray(batch_features))
    return np.asarray(total_features)

''' Function use to perform hot encoding of labels according to the different classes'''
def encodeLabels(total_labels, num_batches, num_images):
    encoded_labels = []
    y = np.zeros((4))
    for i in range(0, num_batches):
        encoded_batches = []
        for j in range(0, num_images):
            x = copy.deepcopy(y)
            x[int(total_labels[i][j])-1] = int(1)
            encoded_batches.append(x)
        #del encoded_batches[-1]
        encoded_labels.append(np.asarray(encoded_batches))
    return np.asarray(encoded_labels)

#Helper function created to output a dictionary used to store the values associated with a particular class
def classOutputs(num_classes):
    classes = []
    outputs = []
    for a in range(0,num_classes):
        classes.append(a)
        outputs.append([])
    return dict(zip(classes, outputs))

#Helper function to add bias term '1' to the datapoint
def addBias(features, num_batches, num_images):
    new_features = []
    for i in range(0, num_batches):
        batches = []
        for j in range(0, num_images):
            data_point = np.insert(features[i][j],512,1)
            batches.append(np.reshape(data_point,(513,1)))
        new_features.append(batches)
    return np.asarray(new_features)

#Helper function to normalize the data:
def normalizeData(features, num_batches, num_images):
    for i in range(0, num_batches):
        for j in range(0, num_images):
            max_x = np.max(features[i][j])
            min_x = np.min(features[i][j])
            for k in range(0, len(features[i][j])):
                a = np.subtract(features[i][j][k], min_x)
                b = np.subtract(max_x, min_x)
                features[i][j][k] = np.divide(a,b)
    return features

#Helper function to standardize the data:
def standardizeData(features, num_batches, num_images):
    for i in range(0, num_batches):
        for j in range(0, num_images):
            mean_x = np.mean(features[i][j])
            std_x = np.std(features[i][j])
            for k in range(0, len(features[i][j])):
                a = np.subtract(features[i][j][k], mean_x)
                features[i][j][k] = np.divide(a,std_x)
    return features

def l2normData(features, num_batches, num_images):
    for i in range(0, num_batches):
        for j in range(0, num_images):
            l2norm = np.linalg.norm(features[i][j])
            features[i][j] = np.divide(features[i][j], l2norm)
    return features

#Helper function to calculate the log likelihood
'''def logLikelihood(weights, features, num_batches, num_images):
    log_likelihood = []
    for i in range(0, num_batches):
        for j in range(0, num_images):
            numerator = np.exp(np.matmul(weights, total_features[i][j]))
            denominator = np.sum(np.exp(np.matmul(weights, total_features[i][j])))
            sigmoid = np.divide(numerator,(denominator))
            log_likelihood.append(np.log(sigmoid))
    log_likelihood = np.asarray(log_likelihood)
    return np.mean(log_likelihood, axis=0)
'''

def logLikelihood(weights, features, num_batches, num_images, encoded_labels):
    log_likelihood = []
    weights_k_1 = weights[0:3][:]
    for i in range(0, num_batches):
        for j in range(0, num_images):
            index = np.argmax(encoded_labels[i][j])
            if index == 3:
                continue
            else:
                first_term = np.sum(np.matmul(weights[index], features[i][j]))
                second_term = np.log(np.sum((np.exp(np.matmul(weights_k_1, features[i][j])))))
                diff = second_term - first_term
                log_likelihood.append(diff)
    log_likelihood = np.asarray(log_likelihood)
    value = np.mean(log_likelihood)
    return value

#Helper function to get the accuracy
def getPredictions(weights, num_batches, num_images, encoded_labels, features):
    predicted = []
    true = []
    for batch in range(0, num_batches):
        for image in range(0, num_images):
                index = np.argmax(encoded_labels[batch][image])
                true.append(index + 1)
                numerator = np.matmul(weights, total_features[batch][image])
                #print(numerator)
                denominator = np.sum(np.exp(np.matmul(weights, total_features[batch][image])))
                sigmoid = np.divide(numerator,(1+denominator))
                pred_label = np.argmax(sigmoid)+1
                predicted.append(pred_label)
    accuracy = accuracy_score(true, predicted)
    return accuracy, true, predicted

#Evaluate performance on training data
#Load the training data
index_pairs = []
first_indices = []
image_keys = []
image_IDs = []
image_labels = []
num_batches = 250
''' Reading the Train_Features.pkl file using pickle'''
with open('Train_Features.pkl', 'rb') as f:
	train_features = pickle.load(f, encoding = "latin1")

''' Reading the Train_Labels.csv file using pandas'''
train_labels = pandas.read_csv('Train_Labels.csv')


''' train_features is a dictionary of the image name and the corresponding 
	features. Therefore we extract the key values from this dictionary as this
	is what we will be using while splitting the data into batches. '''
train_image_names = list(train_features.keys())
train_num_images = int(len(train_image_names)/num_batches)
train_image_keys = getBatches(train_image_names, train_num_images, num_batches)
total_labels = getLabels(train_labels, train_image_names, train_image_keys, train_num_images)
features = getFeatures(train_features, num_batches, train_image_keys, train_num_images)
encoded_labels = encodeLabels(total_labels, num_batches, train_num_images)

train_features = normalizeData(features, num_batches, train_num_images)
print("Data Normalized!")
total_features = addBias(train_features, num_batches, train_num_images)

print("Data Ready!")
print(total_features[0].shape)

# Type 1 Implementation
''' Initialize the weight vector'''
mu = 0
sigma = 0.1
np.random.seed(2)
weights = np.random.normal(mu, sigma, (4,513))
#weights = np.zeros((4,513))
weights_k_1 = weights[0:3][:]
[m,n] = weights.shape
max_epoch = 1000
n_0 = 0.1
n_1 = 1
#eta = 0.1
delta = 0.00001
log_likelihoods_all = []
epochs = []
print(weights.shape)

class_outputs = classOutputs(m)
print("Weights before: ", weights)
outputs = []
train_accuracy = []

for epoch in range(0, max_epoch):
    eta = n_0/(n_1 + epoch)
    print("Epoch:", epoch)
    print("Learning rate: ", eta)
    epochs.append(epoch)
    if epoch > 0:
        #print("L_theta_new in elif loop: ", L_theta_new)
        L_theta_old = copy.deepcopy(L_theta_new)
        #print("L_theta_old: ", L_theta_old)
    for batch in range(0, num_batches):
        class_outputs_new = copy.deepcopy(class_outputs)
        for image in range(0, train_num_images):
            index = np.argmax(encoded_labels[batch][image])
            if index == 3:
                numerator = np.exp(np.matmul(weights,total_features[batch][image]))
                denominator = np.sum(np.exp(np.matmul(weights_k_1, total_features[batch][image])))
                sigmoid = np.divide(1,(1+denominator))
            else:
                numerator = np.exp(np.matmul(weights,total_features[batch][image]))
                denominator = np.sum(np.exp(np.matmul(weights_k_1, total_features[batch][image])))
                sigmoid = np.divide(numerator,(1+denominator))
                output = np.multiply((1 - sigmoid[index]), total_features[batch][image]) 
                output = np.reshape(output, (1,513))
            class_outputs_new[index].append(output)
        for class_label in range(0,m):
            if not class_outputs_new[class_label]:
                continue
            gradients = np.asarray(class_outputs_new[class_label])
            gradient = np.mean(gradients, axis=0)
            weights[class_label] = np.add(weights[class_label], np.multiply(eta,gradient))
    
    accuracy_train, train_true_labels, train_predicted_labels = getPredictions(weights, num_batches, train_num_images, encoded_labels, total_features)        
    train_accuracy.append(accuracy_train)        
    L_theta_new = logLikelihood(weights, total_features, num_batches, train_num_images, encoded_labels)
    print("loss: ", L_theta_new)
    log_likelihoods_all.append(L_theta_new)
    if epoch>0:
        print("Going in loop")
        if L_theta_new > (1-delta)*L_theta_old:
            break
        else:
            continue
    print("L_theta_new at the end!: ", L_theta_new)

print("Weights after: ", weights)

# Evaluating the performance on validation data:
#Reading the validation data:
index_pairs = []
first_indices = []
image_keys = []
image_IDs = []
image_labels = []
num_batches = 250
new_index = []

with open('Val_Features.pkl','rb') as f:
    val_features = pickle.load(f, encoding = "latin1")
    
val_labels = pandas.read_csv('Val_Labels.csv')

val_image_names = list(val_features.keys())
val_num_images = int(len(val_image_names)/num_batches)
val_image_keys = getBatches(val_image_names, val_num_images, num_batches)
total_labels = getLabels(val_labels, val_image_names, val_image_keys, val_num_images)
features = getFeatures(val_features, num_batches, val_image_keys, val_num_images)
val_encoded_labels = encodeLabels(total_labels, num_batches, val_num_images)

val_features = normalizeData(features, num_batches, val_num_images)
print("Data Normalized!")
val_total_features = addBias(val_features, num_batches, val_num_images)

print("Data Ready!")
print(val_total_features.shape)
print(val_total_features[0].shape)

# Type 1 Implementation
''' Initialize the weight vector'''
mu = 0
sigma = 0.1
np.random.seed(2)
weights_val = np.random.normal(mu, sigma, (4,513))
#weights = np.zeros((4,513))
weights_val_k_1 = weights_val[0:3][:]
[m,n] = weights.shape
max_epoch = 1000
n_0 = 0.1
n_1 = 1
#eta = 0.1
delta = 0.00001
log_likelihoods_val = []
epochs_val = []
print(weights_val.shape)

class_outputs = classOutputs(m)
print("Weights before: ", weights_val)
outputs = []
val_accuracy = []

for epoch in range(0, max_epoch):
    #eta = n_0/(n_1 + epoch)
    print("Epoch:", epoch)
    print("Learning rate: ", eta)
    epochs_val.append(epoch)
    if epoch > 0:
        #print("L_theta_new in elif loop: ", L_theta_new)
        L_theta_old = copy.deepcopy(L_theta_val)
        #print("L_theta_old: ", L_theta_old)
    for batch in range(0, num_batches):
        class_outputs_val = copy.deepcopy(class_outputs)
        for image in range(0, val_num_images):
            index = np.argmax(val_encoded_labels[batch][image])
            if index == 3:
                numerator = np.exp(np.matmul(weights_val,val_total_features[batch][image]))
                denominator = np.sum(np.exp(np.matmul(weights_val_k_1, val_total_features[batch][image])))
                sigmoid = np.divide(1,(1+denominator))
            else:
                numerator = np.exp(np.matmul(weights_val,val_total_features[batch][image]))
                denominator = np.sum(np.exp(np.matmul(weights_val_k_1, val_total_features[batch][image])))
                sigmoid = np.divide(numerator,(1+denominator))
                output = np.multiply((1 - sigmoid[index]), val_total_features[batch][image]) 
                output = np.reshape(output, (1,513))
            class_outputs_val[index].append(output)
        for class_label in range(0,m):
            if not class_outputs_val[class_label]:
                continue
            gradients = np.asarray(class_outputs_val[class_label])
            gradient = np.mean(gradients, axis=0)
            weights_val[class_label] = np.add(weights_val[class_label], np.multiply(eta,gradient))

    accuracy_val, val_true_labels, val_predicted_labels = getPredictions(weights_val, num_batches, val_num_images, val_encoded_labels, val_total_features)        
    val_accuracy.append(accuracy_val)        
    L_theta_val = logLikelihood(weights_val, val_total_features, num_batches, val_num_images, val_encoded_labels)
    print("loss: ", L_theta_val)
    log_likelihoods_val.append(L_theta_val)
    if epoch>0:
        print("Going in loop")
        if L_theta_val > (1-delta)*L_theta_old:
            break
        else:
            continue
    print("L_theta_new at the end!: ", L_theta_val)

print("Weights after: ", weights_val)

#Plotting
plt.xlabel('Epochs')
plt.ylabel('Loss')
train = plt.plot(epochs, log_likelihoods_all,'b')
val = plt.plot(epochs_val, log_likelihoods_val,'r')
train_patch = mpatches.Patch(color='blue', label = 'Training data')
val_patch = mpatches.Patch(color='red', label = 'Validation data' )
plt.legend(handles = [train_patch, val_patch], loc=1, fontsize='small', fancybox=True)
plt.title('Comparison')
plt.show
plt.savefig('plot_2_3_3_a.png')

#Plotting Accuracy compairision
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
train = plt.plot(epochs, train_accuracy,'b')
val = plt.plot(epochs_val, val_accuracy,'r')
train_patch = mpatches.Patch(color='blue', label = 'Training data')
val_patch = mpatches.Patch(color='red', label = 'Validation data' )
plt.legend(handles = [train_patch, val_patch], loc=1, fontsize='small', fancybox=True)
plt.title('Comparison accuracy')
plt.show
plt.savefig('plot_2_3_3_accuracy.png')