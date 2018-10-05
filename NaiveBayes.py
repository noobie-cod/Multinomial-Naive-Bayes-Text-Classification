from pprint import pprint
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier

#filename = 'Data/yelp_labelled.txt'
filename = 'Data/imdb_labelled.txt'

lines = [line.rstrip('\n') for line in open(filename)]

rows = []
for line in lines:
	sentence = line.split("#")
	rows.append(tuple(sentence))

#Spliting the dataset into 75% Training & 25% Testing
train, test = train_test_split(rows, test_size = 0.25)

#Training the Classifier for the yelp Dataset.
model = NaiveBayesClassifier(train)
print(model)

#Accuracy of the Model
print(model.accuracy(test))
