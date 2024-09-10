import pandas as pd
import pickle
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

messages = pd.read_csv('C:/Users/susha/OneDrive/Desktop/Sentiment Analysis/datasets/train.csv', sep=',', names=['label', 'message'])

lem = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()
y = LabelEncoder().fit_transform(messages['label'])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 45)


model0 = LogisticRegression()
model0.fit(xtrain,ytrain)
ypred = model0.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy) 

model1 = MultinomialNB().fit(xtrain, ytrain)
ypred = model1.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy) 

model2 = SVC().fit(xtrain,ytrain)
ypred = model2.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy) 

model3 = KNeighborsClassifier(n_neighbors=8).fit(xtrain,ytrain)
ypred = model3.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy) 

model4 = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)
ypred = model4.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy) 


model5 = xgb.XGBClassifier().fit(xtrain,ytrain)
ypred = model5.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy)

pickle.dump(model4, open("./models/Model.pkl", 'wb'))
pickle.dump(cv, open("./models/cv.pkl", 'wb'))