import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

# List of Genres learnt from the Training Data
genre_list = [
    'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 
    'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music', 
    'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 
    'sport', 'talk-show', 'thriller', 'war', 'western'
]

# Define a fallback genre for movies which the model finds very hard to predict
fallback_genre = 'Unknown'

# Specify the path to the training and test data files

train_data_path = 'C:/Users/cacha/OneDrive/Desktop/Internship assignment/Movie_genre/train_data.txt'
test_data_path = 'C:/Users/cacha/OneDrive/Desktop/Internship assignment/Movie_genre/test_data.txt'

# Ensure the paths are correct and the files exist
if not os.path.isfile(train_data_path):
    raise FileNotFoundError(f"The training data file '{train_data_path}' was not found.")
if not os.path.isfile(test_data_path):
    raise FileNotFoundError(f"The test data file '{test_data_path}' was not found.")

# Load the Training dataset
try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv(train_data_path, sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")
    raise

# Data preprocessing for training data
X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features

# Fit and transform the training data with progress bar
with tqdm(total=50, desc="Vectorizing Training Data") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)

# Train a MultiOutput Naive Bayes classifier using the training data
with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier(naive_bayes)
    multi_output_classifier.fit(X_train_tfidf, y_train)
    pbar.update(50)

# Load the Test dataset
try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv(test_data_path, sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")
    raise

# Data preprocessing for test data
X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())

# Transform the test data with progress bar
with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    pbar.update(50)

# Predict genres on the test data
with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)

# Create a DataFrame for test data with movie names and predicted genres
test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': predicted_genres})

# Replace empty unpredicted genres with the fallback genre
test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)

# Write the results to an output text file with proper formatting
with open("model_evaluation.txt", "w", encoding="utf-8") as output_file:
    for _, row in test_results.iterrows():
        movie_name = row['MOVIE_NAME']
        genre_str = ', '.join(row['PREDICTED_GENRES'])
        output_file.write(f"{movie_name} ::: {genre_str}\n")

# Calculate evaluation metrics using training labels (as a proxy)
y_train_pred = multi_output_classifier.predict(X_train_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='micro')
recall = recall_score(y_train, y_train_pred, average='micro')
f1 = f1_score(y_train, y_train_pred, average='micro')

# Append the evaluation metrics to the output file
with open("model_evaluation.txt", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nModel Evaluation Metrics:\n")
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1-score: {f1:.2f}\n")

print("Model evaluation results and metrics have been saved to 'model_evaluation.txt'.")
