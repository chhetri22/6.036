from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt


def get_stopwords(filename):
    with open(filename) as f:
        stopwords = set([])
        for line in f:
            stopwords.add(line.strip())

    return stopwords

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    
    total_loss = 0
    for x in range(len(labels)):
        #print (feature_matrix[x],theta,labels[x],theta_0)
        if labels[x]*(theta.dot(feature_matrix[x])+theta_0) <= 1:
            current_loss = max(0,1-(labels[x]*(theta.dot(feature_matrix[x])+theta_0)))
            total_loss += current_loss

    
    return float(total_loss)/len(labels)

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    #print (current_theta,feature_vector)

    if label*(np.dot(current_theta,feature_vector)+current_theta_0) <= 0:
        new_theta = np.add(current_theta,np.multiply(label,feature_vector))
        new_theta_0 = current_theta_0 + label
        return (new_theta,new_theta_0)
    else:
        return (current_theta, current_theta_0)

def perceptron(feature_matrix, labels, T):
    """
    Section 1.4a
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    for t in range(T):
        for i in range(len(feature_matrix)):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            
    return (theta,theta_0)
    
    
def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.4b
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta = np.zeros(len(feature_matrix[0]))
    theta_sum = np.array([0 for i in range(len(feature_matrix[0]))])
    theta_0 = 0
    theta_0_sum = 0
    for t in range(T):
        for i in range(len(feature_matrix)):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum = np.add(theta_sum,theta)
            theta_0_sum += theta_0
    
    iterations = len(feature_matrix)*T         
    average_theta = np.divide(theta_sum,iterations)
    average_theta_0 = np.divide(theta_0_sum,iterations)
    
    
    return (average_theta,average_theta_0)

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(current_theta.dot(feature_vector)+current_theta_0) <= 1:
        new_theta = (1-L*eta)*current_theta+np.multiply(eta*label,feature_vector)
        new_theta_0 = (1-L*eta)*current_theta_0 + eta*label
    else:
        new_theta = (1-L*eta)*current_theta
        new_theta_0 = (1-L*eta)*current_theta_0
        
    return (new_theta,new_theta_0)

def pegasos(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    
    For each update, set learning rate = 1/sqrt(t), 
    where t is a counter for the number of updates performed so far (between 1 
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = np.zeros(len(feature_matrix[0]))
    theta_0 = 0
    count = 0

    for t in range(T):


        i_vals = list(range(len(feature_matrix)))
        np.random.shuffle(i_vals)
        for i in i_vals:
            count += 1
            eta = 1/(1.5*count**0.5)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
            
    return (theta,theta_0)

### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    classifications = np.zeros(len(feature_matrix))
    for i in range(len(feature_matrix)):
        if theta.dot(feature_matrix[i])+theta_0 > 0:
            classifications[i] = 1
        else:
            classifications[i] = -1
    return classifications

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta,theta_0 = perceptron(train_feature_matrix, train_labels, T)
    classifications = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(classifications,train_labels)
    
    classifications_validation = classify(val_feature_matrix, theta, theta_0)
    validation_accuracy = accuracy(classifications_validation,val_labels)
        
    return (train_accuracy,validation_accuracy)

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta,theta_0 = average_perceptron(train_feature_matrix, train_labels, T)
    #print (theta,theta_0)
    classifications = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(classifications,train_labels)
    
    classifications_validation = classify(val_feature_matrix, theta, theta_0)
    validation_accuracy = accuracy(classifications_validation,val_labels)
        
    return (train_accuracy,validation_accuracy)

def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Section 2.9
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta,theta_0 = pegasos(train_feature_matrix, train_labels, T, L)
    classifications = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(classifications,train_labels)
    
    classifications_validation = classify(val_feature_matrix, theta, theta_0)
    validation_accuracy = accuracy(classifications_validation,val_labels)
        
    return (train_accuracy,validation_accuracy)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

# def bag_of_words(texts):
#     """
#     Inputs a list of string reviews
#     Returns a dictionary of unique unigrams occurring over the input

#     Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
#     """
#     stopwords = get_stopwords("stopwords.txt")
#     freq_dictionary = {}
#     dictionary = {} # maps word to unique index
#     for text in texts:
#         word_list = extract_words(text)
#         for word in word_list:
#             if word not in stopwords:
#                 if word not in dictionary: 
#                     dictionary[word] = len(dictionary)
#     return dictionary

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    stopwords = get_stopwords("stopwords.txt")
    freq_dictionary = {}
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in stopwords:
                if word not in dictionary:
                    if word not in freq_dictionary:
                        freq_dictionary[word] = 1
                    else:
                        freq_dictionary[word] += 1
                        if freq_dictionary[word] == 2:
                            dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

def extract_additional_features(reviews):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """
    return np.ndarray((len(reviews), 0))

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    #print (len(preds),len(targets))
    return (preds == targets).mean()

