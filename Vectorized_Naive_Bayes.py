from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#filename = 'Data/yelp_labelled.txt'
#filename = 'Data/imdb_labelled.txt'
filename= 'Data/amazon.txt'

lines = [line.rstrip('\n') for line in open(filename)]

rows = []
for line in lines:
	sentence = line.split("#")
	rows.append(tuple(sentence))

#Spliting the dataset into 75% Training & 25% Testing
train, test = train_test_split(rows, test_size = 0.25)

#X -> Text Data , y -> Labels
X_train = []
X_test = []
y_train = [] 
y_test = []
for i in range(0,len(train)):
	X_train.append(train[i][0])
	y_train.append(train[i][1])
for i in range(0,len(test)):
	X_test.append(test[i][0])
	y_test.append(test[i][1])

#instantiate the vectorizer to create Vocabulary
vect = CountVectorizer()

#learn training data vocabulary, then use it to create a document-term matrix theb -> fit
vect.fit(X_train)

#Transform Training Data, we get document-term matrix
X_train_dtm = vect.fit_transform(X_train)

#Transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)

#instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

#Training the model
nb.fit(X_train_dtm, y_train)

#make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

#calculate accuracy of class predictions
print("Accuracy : ",metrics.accuracy_score(y_test, y_pred_class))

#Print the confusion Matrix
print("Confusion Matrix : \n",metrics.confusion_matrix(y_test, y_pred_class))


