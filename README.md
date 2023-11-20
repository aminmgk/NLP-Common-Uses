Simplified examples using Python and popular libraries like scikit-learn and TensorFlow for sentiment analysis (tweak parameters depending on the case):

```python
# Sentiment Analysis Example

# Step 1: Data Collection
# Assume you have a dataset with 'text' and 'sentiment' columns.

# Step 2: Text Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace this with your dataset)
texts = ["I love this product!", "Not satisfied with the service.", "Neutral review."]
sentiments = ["positive", "negative", "neutral"]

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
```

For text summarization, you might use a library like `gensim`:

```python
# Text Summarization Example

# Assume you have a dataset with 'document' and 'summary' columns.

from gensim.summarization import summarize

# Sample data (replace this with your dataset)
document = "This is a long document. It contains multiple sentences and information."

# Generate summary
summary = summarize(document)

print("Original Document:\n", document)
print("\nSummary:\n", summary)
```

For speech recognition, you could use a library like `SpeechRecognition`:

```python
# Speech Recognition Example

import speech_recognition as sr

# Step 1: Data Collection (assuming you have audio files)
# Step 2: Audio Preprocessing is often done by the library

# Step 3: Feature Extraction
recognizer = sr.Recognizer()

# Sample data (replace this with your audio file)
audio_file_path = "path/to/audio/file.wav"

with sr.AudioFile(audio_file_path) as source:
    audio_data = recognizer.record(source)

# Step 4: Model Selection & Training
text_transcription = recognizer.recognize_google(audio_data)

# Step 5: Evaluation
print("Transcription:\n", text_transcription)
```
