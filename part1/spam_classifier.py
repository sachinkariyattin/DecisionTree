"""
Problem formulation:
The given data set has to be classified into SPAM and NOT SPAM messages using two types of classifiers, Bayes and
Decision Tree. Initial step for both the models is to read each of the messages line by line and store the words in a
data structure. Some of the stop words as well as email-headers are cleaned up during the data pre processing stage.

Bayes Classifier:
Training:
The email messages in the training set are read one by one and the proabability of P(w|S=1) and P(w|S=0)(SPAM and NOT SPAM)
are estimated both for binary and continuous features. Binary features are calculated based on the presence of word in
the given message, i.e ex: if the word 'offer' appears in an email, then it is considered as 1 (i.e duplicates are not
considered). Continuous features are considered based on the number of times the word occurs in the message divided by the
total number of words.
Each of these probabilities P(w/S=1),P(w/s=0) for binary and P(w/S=1), P(w/S=0) for continuous features are stored in the
model file after the training. The top 10 words with the P(S=1/w) and P(S=0/w) are computed using these probability
distribution tables.

Testing:
In the testing phase, the probability distribution tables in the model file are retrieved. Now,  a test email
is read word by word and with the help of variable elimination and the probabilities in the model file, the probability
of Spam given all the words in the email is calculated, i.e P(S=1/w1,w2...wn). Also P(S=1/w1,w2...wn) is calculated.
If the probability of document being SPAM is greater than the probability of document being NOT SPAM, then it is classified
as SPAM OR NOT SPAM otherwise.

The results obtained for both the features are given below

-------------------------------------------------------------------------------------------------------------
CONFUSION MATRIX FOR BINARY FEATURES
Accuracy = 92.6389976507%

			Predicted Spam		Predicted Not Spam
Actual Spam			1037			148
Actual Not Spam		40			1329
-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------
CONFUSION MATRIX FOR CONTINUOUS FEATURES
Accuracy = 80.9710258418%

			Predicted Spam		Predicted Not Spam
Actual Spam			765			420
Actual Not Spam		66			1303
-------------------------------------------------------------------------------------------------------------

Decision Tree:
Training:
All the email messages are read and the data is stored in the form of a dictionary in the following format
            WORD1   WORD2   WORD3........................................WORDn
Filename    freq1   freq1   freq3                                        freqn

Decision tree is constructed based on the entropy based greedy algorithm for both the features(continuous and binary).
For binary features, the tree is split if given word exists in the message. For continuous features, the tree is split
if the frequency of the word is more than 2.

At the end of the training, two decision trees, one for binary features and the other for continuous features are stored
in a model file (stored in pickle file). At the end of training, the computed decision tree is displayed on the console

Testing:
Both the tree objects from the model file are retrieved during the testing. A test email is passed through the decision tree
and classified as SPAM OR NOTSPAM.

The results obtained for both the features are given below
DECISION TREE CLASSIFIER
-------------------------------------------------------------------------------------------------------------
CONFUSION MATRIX FOR BINARY FEATURES
Accuracy = 53.602192639%

			Predicted Spam		Predicted Not Spam
Actual Spam			0			1185
Actual Not Spam		0			1369
-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------
CONFUSION MATRIX FOR CONTINUOUS FEATURES
Accuracy = 96.12%

			Predicted Spam		Predicted Not Spam
Actual Spam			1138			47
Actual Not Spam		58			1311
-------------------------------------------------------------------------------------------------------------


As it can be seen from the above results, Bayes classifier works well for binary features, whereas the accuracy is slightly
less for the continuous features.

Decision tree accuracy for binary features is very low but the continuous feature classification accuracy is high.

Overall, decision tree classifier for continuous features gives the best results
"""


from __future__ import division
import os
from collections import defaultdict
from math import log
import math
import cPickle as pickle
import sys


# Node Class
class decisionnode:
  def __init__(self,col=-1,results=None,tb=None,fb=None):
    self.col=col
    self.results=results
    self.tb=tb
    self.fb=fb

# Draw confusion matrix
def confusion_matrix(feature, predictedspam_spam_messages, predictedspam_notspam_messages, predictednotspam_spam_messages,
                     predictednotspam_notspam_messages):

    accuracy = (predictedspam_spam_messages + predictednotspam_notspam_messages) / (predictedspam_spam_messages +
                                                                                    predictednotspam_notspam_messages +
                                                                                    predictedspam_notspam_messages +
                                                                                    predictednotspam_spam_messages)
    accuracy = accuracy * 100


    print "-------------------------------------------------------------------------------------------------------------"
    print "CONFUSION MATRIX FOR " + feature
    print "Accuracy = " + str(accuracy) +"%"
    print
    print "\t\t\t" + "Predicted Spam" + "\t\t" + "Predicted Not Spam"
    print "Actual Spam\t\t\t" + str(predictedspam_spam_messages) + "\t\t\t" + str(predictedspam_notspam_messages)
    print "Actual Not Spam\t\t\t" + str(predictednotspam_spam_messages) + "\t\t\t" + str(predictednotspam_notspam_messages)
    print "-------------------------------------------------------------------------------------------------------------"


# Estimate prob using variable elimination
def get_predictions(dir, modelfile_vector, classtype, p_spam, p_notspam):
    cleanup_list = ['to', 'cc', 'subject', 'in-reply-to', 'references', 'mime-version', 'content-type', 'message-id',
                    'x-loop', 'sender', 'errors-to', 'list-id', 'date', 'from', 'the', 'and', 'with', 'by', 'received',
                    'delivered-to',
                    'return-path', 'sender', 'of', 'a', 'your', 'in', 'id', 'is', 'you', 'for', 'that', 'i', 'on', 'it',
                    'localhost', 'oct', 'aug', 'sep', 'this', 'jan', 'feb', 'march', 'april', 'may', 'june', 'july',
                    'nov', 'dec']

    actual_spam_messages = 0

    binpredictedspam_spam_messages = 0
    binpredictedspam_notspam_messages = 0
    binpredictednotspam_spam_messages = 0
    binpredictednotspam_notspam_messages = 0

    ratiopredictedspam_spam_messages = 0
    ratiopredictedspam_notspam_messages = 0
    ratiopredictednotspam_spam_messages = 0
    ratiopredictednotspam_notspam_messages = 0

    for filename in os.listdir(dir):
        actual_spam_messages += 1
        words_in_message = set()
        file = open(dir + "/" + filename, 'r')
        for line in file:
            words_in_message.update(w.lower() for w in line.split())

        # Predict spam or not spam for binary features
        # Check spam prob
        numerator = p_spam
        denominator = p_notspam
        for word in modelfile_vector:
            if word in words_in_message:
                numerator *= modelfile_vector[word]['binspam']
                denominator *= modelfile_vector[word]['binnotspam']

            else:
                numerator *= (1 - modelfile_vector[word]['binspam'])
                denominator *= (1 - modelfile_vector[word]['binnotspam'])


        if numerator == 0:
            predicted_spam_prob = 0
        else:
            predicted_spam_prob = numerator / (numerator + denominator)

        # Check not spam prob
        numerator = p_notspam
        denominator = p_spam
        for word in modelfile_vector:
            if word in words_in_message:
                numerator *= modelfile_vector[word]['binnotspam']
                denominator *= modelfile_vector[word]['binspam']
            else:
                numerator *= (1 - modelfile_vector[word]['binnotspam'])
                denominator *= (1 - modelfile_vector[word]['binspam'])

        if numerator == 0:
            predicted_notspam_prob = 0
        else:
            predicted_notspam_prob = numerator / (numerator + denominator)

        if classtype == 'SPAM':
            if predicted_spam_prob >= predicted_notspam_prob:
                binpredictedspam_spam_messages += 1
            else:
                binpredictedspam_notspam_messages += 1
        else:
            if predicted_spam_prob >= predicted_notspam_prob:
                binpredictednotspam_spam_messages += 1
            else:
                binpredictednotspam_notspam_messages += 1

        # Spam, not spam using ratio features
        numerator = p_spam
        denominator = p_notspam
        for word in modelfile_vector:
            if word in words_in_message:
                numerator *= modelfile_vector[word]['ratiospam']
                denominator *= modelfile_vector[word]['rationotspam']
            else:
                numerator *= (1 - modelfile_vector[word]['ratiospam'])
                denominator *= (1 - modelfile_vector[word]['rationotspam'])

        if numerator == 0:
            predicted_spam_prob = 0
        else:
            predicted_spam_prob = numerator / (numerator + denominator)

        # Check not spam prob
        numerator = p_notspam
        denominator = p_spam
        for word in modelfile_vector:
            if word in words_in_message:
                numerator *= modelfile_vector[word]['rationotspam']
                denominator *= modelfile_vector[word]['ratiospam']
            else:
                numerator *= (1 - modelfile_vector[word]['rationotspam'])
                denominator *= (1 - modelfile_vector[word]['ratiospam'])

        if numerator == 0:
            predicted_notspam_prob = 0
        else:
            predicted_notspam_prob = numerator / (numerator + denominator)

        if classtype == 'SPAM':
            if predicted_spam_prob >= predicted_notspam_prob:
                ratiopredictedspam_spam_messages += 1
            else:
                ratiopredictedspam_notspam_messages += 1
        else:
            if predicted_spam_prob >= predicted_notspam_prob:
                ratiopredictednotspam_spam_messages += 1
            else:
                ratiopredictednotspam_notspam_messages += 1

    if classtype == 'SPAM':
        return binpredictedspam_spam_messages,binpredictedspam_notspam_messages,ratiopredictedspam_spam_messages,ratiopredictedspam_notspam_messages
    else:
        return binpredictednotspam_spam_messages,binpredictednotspam_notspam_messages,ratiopredictednotspam_spam_messages,ratiopredictednotspam_notspam_messages


# Bayes classifier test data
def test_bayes(dir, modelfile):

    spam_path = dir + '/spam'
    notspam_path = dir + '/notspam'
    modelfile_vector = {}
    p_spam = 0
    p_notspam = 0

    # Read the model file
    file = open(modelfile, 'r')
    for line in file:
        word = line.split()
        for i in range(0, len(word)):
            if i == 0:
                if word[i] == 'SPAM':
                    p_spam = float(word[1])
                    p_notspam = float(word[2])
                    break
                else:
                    modelfile_vector[word[i]] = {}
            elif i == 1:
                modelfile_vector[word[0]]['binspam'] = {}
                modelfile_vector[word[0]]['binspam'] = float(word[i])
            elif i == 2:
                modelfile_vector[word[0]]['binnotspam'] = {}
                modelfile_vector[word[0]]['binnotspam'] = float(word[i])
            elif i == 3:
                modelfile_vector[word[0]]['ratiospam'] = {}
                modelfile_vector[word[0]]['ratiospam'] = float(word[i])
            else:
                modelfile_vector[word[0]]['rationotspam'] = {}
                modelfile_vector[word[0]]['rationotspam'] = float(word[i])

    # Predict spam messages and not spam messages
    bin_spam, bin_notspam, ratio_spam, ratio_notspam = get_predictions(spam_path, modelfile_vector, 'SPAM', p_spam, p_notspam)
    binnotspam_spam, binnotspam_notspam, rationotspam_spam, rationotspam_notspam = get_predictions(notspam_path, modelfile_vector, 'NOT SPAM', p_spam, p_notspam)

    confusion_matrix('BINARY FEATURES',bin_spam,bin_notspam,binnotspam_spam,binnotspam_notspam)
    confusion_matrix('CONTINUOUS FEATURES',ratio_spam,ratio_notspam,rationotspam_spam,rationotspam_notspam)


# Bayes train technique
def train_bayes(dir, modelfile):

    spam_path = dir + '/spam'
    notspam_path = dir + '/notspam'
    words_by_spam_file = {}
    all_words_spam_set = set()
    words_by_notspam_file = {}
    all_words_notspam_set = set()
    all_words_spam = []
    all_words_notspam = []

    spam_given_word = {}
    notspam_given_word = {}

    binary_vector = {}
    ratio_vector = {}
    raw_count_spam = defaultdict(int)
    raw_count_notspam = defaultdict(int)

    total_spam_messages = 0
    total_notspam_messages = 0

    # Read all the words in spam directory
    word_count_per_file_spam = defaultdict(int)
    for filename in os.listdir(spam_path):
        total_spam_messages += 1
        words_by_spam_file[filename] = set()
        file = open(spam_path + "/" + filename, 'r')
        for line in file:
            words_by_spam_file[filename].update(w.lower() for w in line.split())
            all_words_spam_set.update(w.lower() for w in line.split())
            all_words_spam += [w.lower() for w in line.split()]
        for word in words_by_spam_file[filename]:
            word_count_per_file_spam[word] += 1

    # Read all the words in not spam directory
    word_count_per_file_notspam = defaultdict(int)
    for filename in os.listdir(notspam_path):
        total_notspam_messages += 1
        words_by_notspam_file[filename] = set()
        file = open(notspam_path + "/" + filename, 'r')
        for line in file:
            words_by_notspam_file[filename].update(w.lower() for w in line.split())
            all_words_notspam_set.update(w.lower() for w in line.split())
            all_words_notspam += [w.lower() for w in line.split()]
        for word in words_by_notspam_file[filename]:
            word_count_per_file_notspam[word] += 1

    # Binary spam and not spam
    total_notspam_words = len(all_words_notspam_set)
    total_spam_words = len(all_words_spam_set)

    for word in all_words_spam_set:
        if word not in binary_vector:
            binary_vector[word] = {}
            binary_vector[word]['spam'] = word_count_per_file_spam[word] / total_spam_messages
            binary_vector[word]['notspam'] = 0.00000001

    for word in all_words_notspam_set:
        if word in binary_vector:
            binary_vector[word]['notspam'] = word_count_per_file_notspam[word] / total_notspam_messages

        else:
            binary_vector[word] = {}
            binary_vector[word]['spam'] = 0.00000001
            binary_vector[word]['notspam'] = word_count_per_file_notspam[word] / total_notspam_messages

    # Ratio vector calculation
    cleanup_list = ['to','cc','subject','in-reply-to','references','mime-version','content-type','message-id','x-loop','sender','errors-to','list-id','date','from','the','and','with','by','received','delivered-to',
                    'return-path','sender','of', 'a',  'your',  'in',  'id', 'is', 'you', 'for','that', 'i','on','it','localhost','oct','aug','sep','this','jan','feb','march','april','may','june','july','nov','dec']

    for word in all_words_spam:
        raw_count_spam[word] += 1

    for word in all_words_notspam:
        raw_count_notspam[word] += 1

    for word in all_words_spam_set:
        if raw_count_spam[word] == 1:
            raw_count_spam.pop(word)
            binary_vector.pop(word)
        elif '<' in word or '>' in word or '=' in word or ':' or '#' in word or word in cleanup_list:
            raw_count_spam.pop(word)
            binary_vector.pop(word)

    for word in all_words_notspam_set:
        if raw_count_notspam[word] == 1:
            raw_count_notspam.pop(word)
            if word in binary_vector:
                binary_vector.pop(word)
        elif '<' in word or '>' in word or '=' in word or ':' in word or word in cleanup_list:
            raw_count_notspam.pop(word)
            if word in binary_vector:
                binary_vector.pop(word)

    for word in raw_count_spam:
        if word not in ratio_vector:
            ratio_vector[word] = {}
            ratio_vector[word]['spam'] = raw_count_spam[word] / total_spam_words
            ratio_vector[word]['notspam'] = 0.00000001

    for word in raw_count_notspam:
        if word in ratio_vector:
            ratio_vector[word]['notspam'] = raw_count_notspam[word] / total_notspam_words
        else:
            ratio_vector[word] = {}
            ratio_vector[word]['spam'] = 0.00000001
            ratio_vector[word]['notspam'] = raw_count_notspam[word] / total_notspam_words

    # Get the max 10 spam and not spam words
    p_spam = total_spam_messages / (total_spam_messages + total_notspam_messages)
    p_notspam = total_notspam_messages / (total_spam_messages + total_notspam_messages)

    for word in ratio_vector:
        spam_given_word[word] = ratio_vector[word]['spam'] * p_spam
        notspam_given_word[word] = ratio_vector[word]['notspam'] * p_notspam / ((ratio_vector[word]['spam'] * p_spam) + (ratio_vector[word]['notspam'] *  p_notspam))

    max_spam_words = sorted(spam_given_word, key=spam_given_word.__getitem__, reverse=True)
    max_notspam_words = sorted(notspam_given_word, key=notspam_given_word.__getitem__, reverse=True)


    # write the data to a model file
    file = open(modelfile, "w+")
    for key in binary_vector:
        file.write(str(key) + " " + str(binary_vector[key]['spam']) + " " + str(binary_vector[key]['notspam']) +  " " + str(ratio_vector[key]['spam']) + " " + str(ratio_vector[key]['notspam']) + "\n")
    file.write("SPAM " + str(p_spam) + " " + str(p_notspam))
    file.close()

    # disp spam and not spam words
    print "SPAM: ",
    for i in range(0, 10):
        print max_spam_words[i] + ", ",

    print "\n"
    print "NOT SPAM: ",
    for i in range(len(max_spam_words) - 1, len(max_spam_words) - 10, -1):
        print max_spam_words[i] + ", ",



# Decision tree technique
def dt_train(dataset_dir, modelfile):
    cleanup_list = ['to', 'cc', 'subject', 'in-reply-to', 'references', 'mime-version', 'content-type', 'message-id',
                    'x-loop', 'sender', 'errors-to', 'list-id', 'date', 'from', 'the', 'and', 'with', 'by', 'received',
                    'delivered-to',
                    'return-path:', 'sender', 'of', 'a', 'your', 'in', 'id', 'is', 'you', 'for', 'that', 'i', 'on', 'it',
                    'localhost', 'oct', 'aug', 'sep', 'this', 'jan', 'feb', 'march', 'april', 'may', 'june', 'july',
                    'nov', 'dec']
    spam_path = dataset_dir + '/spam'
    notspam_path = dataset_dir + '/notspam'

    # Read all the words in spam directory
    all_words_set = set()
    all_words = []
    raw_count_words = defaultdict(int)
    for filename in os.listdir(spam_path):
        file = open(spam_path + "/" + filename, 'r')
        for line in file:
            all_words_set.update(w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list)
            all_words += [w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]


    # Read all the words in not spam directory
    for filename in os.listdir(notspam_path):
        file = open(notspam_path + "/" + filename, 'r')
        for line in file:
            all_words_set.update(w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list)
            all_words += [w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]

    for word in all_words:
        raw_count_words[word] += 1

    for word in all_words_set:
        if raw_count_words[word] <= 30:
            raw_count_words.pop(word)

    # Compute binary  and continuous features
    continuous_vector = {}
    for filename in os.listdir(spam_path):
        word_count_in_file = defaultdict(int)
        all_words_per_file = []
        continuous_vector[filename] = {}
        all_words_per_file_set = set()
        file = open(spam_path + "/" + filename, 'r')
        for line in file:
            all_words_per_file_set.update(w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list)
            all_words_per_file += [w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]
        for word in all_words_per_file:
            word_count_in_file[word] += 1
        continuous_vector[filename]['CLASS'] = 'SPAM'
        for word in all_words_per_file_set:
            if word in raw_count_words:
                continuous_vector[filename][word] = word_count_in_file[word]

    for filename in os.listdir(notspam_path):
        word_count_in_file = defaultdict(int)
        all_words_per_file = []
        all_words_per_file_set = set()
        continuous_vector[filename] = {}
        file = open(notspam_path + "/" + filename, 'r')
        for line in file:
            all_words_per_file_set.update(w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list)
            all_words_per_file += [w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]
        for word in all_words_per_file:
            word_count_in_file[word] += 1
        for word in all_words_per_file_set:
            if word in raw_count_words:
                continuous_vector[filename][word] = word_count_in_file[word]
        continuous_vector[filename]['CLASS'] = 'NOTSPAM'

    return continuous_vector, raw_count_words

# Get classes based on attributes
def divide_based_on_class(continuous_vector, files, word, feature):
    split_based_on_attribute = None
    if feature == 'Binary':  # check if the feature is Binary or Continuous
        split_based_on_attribute = lambda filename: word in continuous_vector[filename]
    else:
        split_based_on_attribute = lambda filename: word in continuous_vector[filename] and continuous_vector[filename][word] >= 2

    # Divide the rows into two sets and return them
    set1 = [filename for filename in files if split_based_on_attribute(filename)]
    set2 = [filename for filename in files if not split_based_on_attribute(filename)]

    return (set1, set2)


# Classified based on splits
def classified_after_split(continuous_vector, set):
    results = {}
    for filename in set:
        r = continuous_vector[filename]['CLASS']
        if r not in results: results[r] = 0
        results[r] += 1
    return results

# Calculate entropy to get the error minimizing node
def entropy(continuous_vector, set):

   log2 = lambda x:log(x)/log(2)
   results = classified_after_split(continuous_vector, set)

   entropy = 0.0
   for r in results.keys():
      p = float(results[r])/len(set)
      entropy = entropy - p * log2(p)
   return entropy

# Build tree
# Reference: http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html
def build_decisiontree(continuous_vector, files, feature, raw_count_words):

    if len(files) == 0:
        return decisionnode()

    current_score = entropy(continuous_vector, files)
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    for word in raw_count_words:
        set1, set2 = divide_based_on_class(continuous_vector, files, word, feature)

        p = len(set1) / len(files)

        gain = current_score - p * entropy(continuous_vector, set1) - (1 - p) * entropy(continuous_vector, set2)
        if gain > best_gain and len(set1) > 0 and len(set2) > 0:
            best_gain = gain
            best_criteria = word
            best_sets = set1, set2

    if best_gain > 0:
        if best_criteria is not None and feature != 'Binary':
            raw_count_words.remove(best_criteria)
        true_branch = build_decisiontree(continuous_vector,best_sets[0],feature,raw_count_words)
        false_branch = build_decisiontree(continuous_vector,best_sets[1],feature,raw_count_words)
        return decisionnode(col=best_criteria,tb=true_branch,fb=false_branch)
    else:
        return decisionnode(results=classified_after_split(continuous_vector,files).keys())

# dt classifier
def classify(observation,tree,feature):
  if tree.results!=None:
    return str(tree.results)
  else:
    v = observation[tree.col]
    branch=None
    if feature == 'Binary':
      if tree.col in observation: branch=tree.tb
      else: branch=tree.fb
    else:
      if tree.col in observation and observation[tree.col] >= 2: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch,feature)

# Test using decision tree
def test_dt(dir, modelfile):
    spam_path = dir + '/spam'
    notspam_path = dir + '/notspam'

    pickle_list = pickle.load( open( modelfile+".p", "rb" ) )
    binary_tree = pickle_list[0]
    continuous_tree = pickle_list[1]

    cleanup_list = ['to', 'cc', 'subject', 'in-reply-to', 'references', 'mime-version', 'content-type', 'message-id',
                    'x-loop', 'sender', 'errors-to', 'list-id', 'date', 'from', 'the', 'and', 'with', 'by', 'received',
                    'delivered-to',
                    'return-path:', 'sender', 'of', 'a', 'your', 'in', 'id', 'is', 'you', 'for', 'that', 'i', 'on',
                    'it',
                    'localhost', 'oct', 'aug', 'sep', 'this', 'jan', 'feb', 'march', 'april', 'may', 'june', 'july',
                    'nov', 'dec']

    binary_spam_notspam = 0
    binary_spam_spam = 0
    cont_spam_notspam = 0
    cont_spam_spam = 0

    binary_notspam_notspam = 0
    binary_notspam_spam = 0
    cont_notspam_notspam = 0
    cont_notspam_spam = 0

    for filename in os.listdir(spam_path):

        dict_words_in_file = defaultdict(int)
        file = open(spam_path + "/" + filename, 'r')
        for line in file:
            all_words = []
            all_words += [w.lower() for w in line.split() if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]
            for word in all_words:
                dict_words_in_file[word] += 1

        # Predict spam for binary features
        result = classify(dict_words_in_file, binary_tree, 'Binary')
        if 'NOTSPAM' in result:
            binary_spam_notspam += 1
        else:
            binary_spam_spam += 1

        # Predict spam for continuous features
        result = classify(dict_words_in_file, continuous_tree, 'Continuous')
        if 'NOTSPAM' in result:
            cont_spam_notspam += 1
        else:
            cont_spam_spam += 1

    for filename in os.listdir(notspam_path):
        dict_words_in_file = defaultdict(int)
        file = open(notspam_path + "/" + filename, 'r')
        for line in file:
            all_words = []
            all_words += [w.lower() for w in line.split()if '<' not in w and '>' not in w and '=' not in w and ':' not in w and '#' not in w and w not in cleanup_list]
            for word in all_words:
                dict_words_in_file[word] += 1

        # Predict non spam for binary features
        result = classify(dict_words_in_file, binary_tree, 'Binary')
        if 'NOTSPAM' in result:
            binary_notspam_notspam += 1
        else:
            binary_notspam_spam += 1

        # Predict not spam for continuous features
        result = classify(dict_words_in_file, continuous_tree, 'Continuous')
        if 'NOTSPAM' in result:
            cont_notspam_notspam += 1
        else:
            cont_notspam_spam += 1

    confusion_matrix('BINARY FEATURES',binary_spam_spam,binary_spam_notspam,binary_notspam_spam,binary_notspam_notspam)
    confusion_matrix('CONTINUOUS FEATURES', cont_spam_spam, cont_spam_notspam, cont_notspam_spam,
                     cont_notspam_notspam)

# Draw the decision tree
def printtree(tree,indent=''):
    if tree.results!=None:
        print str(tree.results)
    else:
        print str(tree.col)+'? '
        # Print the branches
        print indent+'T->',
        printtree(tree.tb,indent+'  ')
        print indent+'F->',
        printtree(tree.fb,indent+'  ')

# Check the parameters (mode, technique and path)
def train_test(mode, technique, dataset_dir, modelfile):
    if mode == 'train' and technique == 'bayes':
        print "TOP 10 WORDS WILL BE DISPLAYED BELOW"
        train_bayes(dataset_dir, modelfile)
    elif mode == 'test' and technique == 'bayes':
        print "BAYES CLASSIFIER"
        print
        test_bayes(dataset_dir,modelfile)
    elif mode == 'train' and technique == 'dt':
        con_vec, raw_count_words = dt_train(dataset_dir, modelfile)
        binary_tree = build_decisiontree(con_vec, list(con_vec.keys()) , 'Binary', list(raw_count_words.keys()))

        continuous_tree = build_decisiontree(con_vec, list(con_vec.keys()), 'Continuous', list(raw_count_words.keys()))
        print "GENERATED TREE"

        printtree(continuous_tree)
        list_pickle = [binary_tree, continuous_tree]
        pickle.dump(list_pickle, open(modelfile+".p", "wb"))
    elif mode == 'test' and technique == 'dt':
        print "DECISION TREE CLASSIFIER"
        test_dt(dataset_dir, modelfile)


mode = sys.argv[1]
technique = sys.argv[2]
dataset_directory = sys.argv[3]
model_file = sys.argv[4]

train_test(mode, technique, dataset_directory, model_file)
