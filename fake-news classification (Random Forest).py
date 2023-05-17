#Nazif ÇELİK
#library to perform necessary operations on the dataset
import pandas as pd
#I will do multithreading in order to use the processor more efficiently 
#and get fast results. I include the necessary library for this.
from multiprocessing import Process

# Loading the dataset
df = pd.read_csv('WELFake_Dataset.csv')

# Removing missing data from the dataset
df.dropna(inplace=True)

# Checking the number of samples in the dataset
print(f"number of samples in the dataset: {len(df)}")

# Show the first five samples of the dataset
print(df.head())

from sklearn.model_selection import train_test_split
#I divided the dataset into train and test parts
title=df['title']
typr=df['label']
X_train, X_test, y_train, y_test = train_test_split(title, typr, test_size=0.25, random_state=42,shuffle=True)


from sklearn.feature_extraction.text import CountVectorizer

# Fitting CountVectorizer on our train data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Vectorizing test data
X_test_vect = vectorizer.transform(X_test)


from sklearn.ensemble import RandomForestClassifier

# Creating the Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100)

# Training the Random Forest classifier on the train set
rfc.fit(X_train_vect, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = rfc.predict(X_test_vect)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy of the model: {accuracy}")

# Generating a classification report
report = classification_report(y_test, y_pred)
print(report)


