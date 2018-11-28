import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#Exploring the dataset:
print(df.columns)
print(df["pets"].value_counts())

#Augmenting the dataset I:
#Starting with Mapping the Pets Column:
likes_cats_mapping = {"likes dogs and likes cats": 1, "likes dogs": 0, "likes dogs and has cats": 1, "has dogs": 0, "has dogs and likes cats": 1, "likes dogs and dislikes cats": 0, "has dogs and has cats": 1, "has cats": 1, "likes cats": 1, "has dogs and dislikes cats": 0, "dislikes dogs and likes cats": 1, "dislikes dogs and dislikes cats": 0, "dislikes cats": 0, "dislikes dogs and has cats": 1, "dislikes dogs": 0}
df["likes_cats"] = df.pets.map(likes_cats_mapping)

has_cats_mapping = {"likes dogs and likes cats": 0, "likes dogs": 0, "likes dogs and has cats": 1, "has dogs": 0, "has dogs and likes cats": 0, "likes dogs and dislikes cats": 0, "has dogs and has cats": 1, "has cats": 1, "likes cats": 0, "has dogs and dislikes cats": 0, "dislikes dogs and likes cats": 0, "dislikes dogs and dislikes cats": 0, "dislikes cats": 0, "dislikes dogs and has cats": 1, "dislikes dogs": 0}
df["has_cats"] = df.pets.map(has_cats_mapping)

likes_dogs_mapping = {"likes dogs and likes cats": 1, "likes dogs": 1, "likes dogs and has cats": 1, "has dogs": 1, "has dogs and likes cats": 1, "likes dogs and dislikes cats": 1, "has dogs and has cats": 1, "has cats": 0, "likes cats": 0, "has dogs and dislikes cats": 1, "dislikes dogs and likes cats": 0, "dislikes dogs and dislikes cats": 0, "dislikes cats": 0, "dislikes dogs and has cats": 0, "dislikes dogs": 0}
df["likes_dogs"] = df.pets.map(likes_dogs_mapping)

has_dogs_mapping = {"likes dogs and likes cats": 0, "likes dogs": 0, "likes dogs and has cats": 0, "has dogs": 1, "has dogs and likes cats": 1, "likes dogs and dislikes cats": 0, "has dogs and has cats": 1, "has cats": 0, "likes cats": 0, "has dogs and dislikes cats": 1, "dislikes dogs and likes cats": 0, "dislikes dogs and dislikes cats": 0, "dislikes cats": 0, "dislikes dogs and has cats": 0, "dislikes dogs": 0}
df["has_dogs"] = df.pets.map(has_dogs_mapping)

#Combining the Essays:
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

all_essays = df[essay_cols].replace(np.nan, '', regex=True)
df["all_essays"] = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
print(df.columns)

#Dropping the NaN's:
df.dropna(subset = ['likes_cats', 'likes_dogs', 'has_cats', 'has_dogs'], inplace=True)

#Performing the Naive Bayes Classifiers:
#Likes Cats:
from sklearn.model_selection import train_test_split
likes_cats_train_data, likes_cats_test_data, likes_cats_train_labels, likes_cats_test_labels = train_test_split(df["all_essays"], df["likes_cats"], test_size = 0.2, random_state = 1)
print(len(likes_cats_train_data))
print(len(likes_cats_test_data))

from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
counter.fit(likes_cats_train_data)
likes_cats_train_counts = counter.transform(likes_cats_train_data)
likes_cats_test_counts = counter.transform(likes_cats_test_data)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(likes_cats_train_counts, likes_cats_train_labels)
likes_cats_predictions = classifier.predict(likes_cats_test_counts)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print(accuracy_score(likes_cats_test_labels, likes_cats_predictions))
print(recall_score(likes_cats_test_labels, likes_cats_predictions))
print(precision_score(likes_cats_test_labels, likes_cats_predictions))
print(f1_score(likes_cats_test_labels, likes_cats_predictions))

#Likes Dogs:
likes_dogs_train_data, likes_dogs_test_data, likes_dogs_train_labels, likes_dogs_test_labels = train_test_split(df["all_essays"], df["likes_dogs"], test_size = 0.2, random_state = 1)
counter = CountVectorizer()
counter.fit(likes_dogs_train_data)
likes_dogs_train_counts = counter.transform(likes_dogs_train_data)
likes_dogs_test_counts = counter.transform(likes_dogs_test_data)
classifier = MultinomialNB()
classifier.fit(likes_dogs_train_counts, likes_dogs_train_labels)
likes_dogs_predictions = classifier.predict(likes_dogs_test_counts)
print(accuracy_score(likes_dogs_test_labels, likes_dogs_predictions))
print(recall_score(likes_dogs_test_labels, likes_dogs_predictions))
print(precision_score(likes_dogs_test_labels, likes_dogs_predictions))
print(f1_score(likes_dogs_test_labels, likes_dogs_predictions))

#Has Cats:
has_cats_train_data, has_cats_test_data, has_cats_train_labels, has_cats_test_labels = train_test_split(df["all_essays"], df["has_cats"], test_size = 0.2, random_state = 1)
counter = CountVectorizer()
counter.fit(has_cats_train_data)
has_cats_train_counts = counter.transform(has_cats_train_data)
has_cats_test_counts = counter.transform(has_cats_test_data)
classifier = MultinomialNB()
classifier.fit(has_cats_train_counts, has_cats_train_labels)
has_cats_predictions = classifier.predict(has_cats_test_counts)
print(accuracy_score(has_cats_test_labels, has_cats_predictions))
print(recall_score(has_cats_test_labels, has_cats_predictions))
print(precision_score(has_cats_test_labels, has_cats_predictions))
print(f1_score(has_cats_test_labels, has_cats_predictions))

#Has Dogs:
has_dogs_train_data, has_dogs_test_data, has_dogs_train_labels, has_dogs_test_labels = train_test_split(df["all_essays"], df["has_dogs"], test_size = 0.2, random_state = 1)
counter = CountVectorizer()
counter.fit(has_dogs_train_data)
has_dogs_train_counts = counter.transform(has_dogs_train_data)
has_dogs_test_counts = counter.transform(has_dogs_test_data)
classifier = MultinomialNB()
classifier.fit(has_dogs_train_counts, has_dogs_train_labels)
has_dogs_predictions = classifier.predict(has_dogs_test_counts)
print(accuracy_score(has_dogs_test_labels, has_dogs_predictions))
print(recall_score(has_dogs_test_labels, has_dogs_predictions))
print(precision_score(has_dogs_test_labels, has_dogs_predictions))
print(f1_score(has_dogs_test_labels, has_dogs_predictions))

#Augmenting the dataset II:
gender_mapping = {"m": 0, "f": 1}
df["gender"] = df.sex.map(gender_mapping)
print(df['gender'].value_counts())

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
print(df['drinks_code'].value_counts())

drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drug_mapping)
print(df['drugs_code'].value_counts())

smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 4, "yes": 3}
df["smokes_code"] = df.smokes.map(smoke_mapping)
print(df['smokes_code'].value_counts())

df["exclamation_count"] = df.apply(lambda x: x['all_essays'].count("!"), axis=1)
df["cleaned_exclamation_count"] = np.where(df["exclamation_count"] > 50, 0, df["exclamation_count"])

df["essay_cat_count"] = df.apply(lambda x: (x['all_essays'].count(" cat ") + x['all_essays'].count(" cat,") + x['all_essays'].count(" cat.") + x['all_essays'].count(" cat!") + x['all_essays'].count(" cats")), axis=1)
print(df["essay_cat_count"].value_counts())

df["essay_dog_count"] = df.apply(lambda x: (x['all_essays'].count(" dog ") + x['all_essays'].count(" dog,") + x['all_essays'].count(" dog.") + x['all_essays'].count(" dog!") + x['all_essays'].count(" dogs")), axis=1)
print(df["essay_dog_count"].value_counts())

df["essay_fun_count"] = df.apply(lambda x: (x['all_essays'].count(" fun ") + x['all_essays'].count(" fun,") + x['all_essays'].count(" fun.") + x['all_essays'].count(" fun!") + x['all_essays'].count(" fun:")), axis=1)
print(df["essay_fun_count"].value_counts())

df["essay_work_count"] = df.apply(lambda x: (x['all_essays'].count(" work ") + x['all_essays'].count(" work,") + x['all_essays'].count(" work.") + x['all_essays'].count(" work!") + x['all_essays'].count(" work:")), axis=1)
print(df["essay_work_count"].value_counts())

df["essay_len"] = df.apply(lambda x: len(x['all_essays']), axis=1)
print(df["essay_len"].value_counts())

ed_level_mapping = {"graduated from college/university": 2, "graduated from masters program": 3, "working on college/university": 1.5, "working on masters program": 2.5, "graduated from two-year college": 1.5, "graduated from high school": 1, "graduated from ph.d program": 4, "graduated from law school": 3, "working on two-year college": 1, "dropped out of college/university": 1.5, "working on ph.d program": 3.5, "college/university": 2, "graduated from space camp": 0.5, "dropped out of space camp": 0.5, "graduated from med school": 3, "working on space camp": 0.5, "working on law school": 2.5, "two-year college": 1.5, "working on med school": 2.5, "dropped out of two-year college": 1, "dropped out of masters program": 2, "masters program": 3, "dropped out of ph.d program": 3, "dropped out of high school": 0, "high school": 1, "working on high school": 0.5, "space camp": 0.5, "ph.d program": 4, "law school": 3, "dropped out of law school": 2, "dropped out of med school": 2, "med school": 3}
df["ed_level_code"] = df.education.map(ed_level_mapping)
print(df["ed_level_code"].value_counts())

#Dropping the Na's:
feature_data = df[['age', 'income', 'gender', 'ed_level_code', 'drinks_code', 'drugs_code', 'smokes_code', 'likes_cats', 'likes_dogs', 'has_cats', 'has_dogs', 'essay_cat_count', 'essay_dog_count', 'cleaned_exclamation_count', 'essay_fun_count', 'essay_work_count', 'essay_len']]
feature_data.isna().any()

feature_data.dropna(inplace=True)

feature_data.isna().any()

#Correlation analysis on the feature data:
feature_data.corr()

#KNeighbors Classifier:
from sklearn import preprocessing

knn_feature_data = feature_data[['age', 'income', 'ed_level_code', 'gender', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']]
x = knn_feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

knn_feature_data = pd.DataFrame(x_scaled, columns=knn_feature_data.columns)


from sklearn.model_selection import train_test_split

knn_labels = feature_data['likes_cats']
knn_data = knn_feature_data
knn_train_data, knn_test_data, knn_train_labels, knn_test_labels = train_test_split(knn_data, knn_labels, test_size = 0.2, random_state = 1)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 101)
classifier.fit(knn_train_data, knn_train_labels)
print(classifier.score(knn_test_data, knn_test_labels))

import matplotlib.pyplot as plt

scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(knn_train_data, knn_train_labels)
    scores.append(classifier.score(knn_test_data, knn_test_labels))
    
plt.plot(range(1,200), scores)
plt.xlabel('k values')
plt.ylabel('accuracy score')
plt.title('Accuracy score at various values for k')
plt.show()


#Multiple Linear Regressor:
from sklearn import preprocessing

mlr_feature_data = feature_data[['income', 'ed_level_code', 'gender', 'drinks_code', 'drugs_code', 'smokes_code', 'likes_cats', 'likes_dogs', 'has_cats', 'has_dogs', 'cleaned_exclamation_count', 'essay_len' ]]
x = mlr_feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

mlr_feature_data = pd.DataFrame(x_scaled, columns=mlr_feature_data.columns)

from sklearn.model_selection import train_test_split

mlr_labels = feature_data['age']
mlr_data = mlr_feature_data
mlr_X_train, mlr_X_test, mlr_y_train, mlr_y_test = train_test_split(mlr_data, mlr_labels, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(mlr_X_train,mlr_y_train)

model.score(mlr_X_train,mlr_y_train)

model.score(mlr_X_test,mlr_y_test)

mlr_y_predicted = model.predict(mlr_X_test)

plt.scatter(mlr_y_test,mlr_y_predicted)
plt.xlabel('Age')
plt.ylabel('Predicted Age')
plt.xlim(15,75)
plt.ylim(15,75)
plt.show()

#KNeighbors Regressor:
from sklearn import preprocessing

knr_feature_data = feature_data[['income', 'ed_level_code', 'gender', 'drinks_code', 'drugs_code', 'smokes_code', 'likes_cats', 'likes_dogs', 'has_cats', 'has_dogs', 'cleaned_exclamation_count', 'essay_len' ]]
x = knr_feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

knr_feature_data = pd.DataFrame(x_scaled, columns=knr_feature_data.columns)

from sklearn.model_selection import train_test_split

knr_labels = feature_data['age']
knr_data = knr_feature_data
knr_X_train, knr_X_test, knr_y_train, knr_y_test = train_test_split(knr_data, knr_labels, test_size = 0.2, random_state = 1)

from sklearn.neighbors import KNeighborsRegressor
classifier = KNeighborsRegressor(n_neighbors = 27)
classifier.fit(knr_X_train, knr_y_train)
print(classifier.score(knr_X_test, knr_y_test))

import matplotlib.pyplot as plt

scores = []
for k in range(1, 200):
    classifier = KNeighborsRegressor(n_neighbors = k)
    classifier.fit(knr_X_train, knr_y_train)
    scores.append(classifier.score(knr_X_test, knr_y_test))
    
plt.plot(range(1,200), scores)
plt.xlabel('k values')
plt.ylabel('accuracy score')
plt.title('Accuracy score at various values for k')
plt.show()
