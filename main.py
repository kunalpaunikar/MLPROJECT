import nltk
import numpy as np
import random
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load intents JSON file
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    return ' '.join(tokens)

# Preprocess patterns and prepare training data
sentences = []
labels = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        processed_text = preprocess_text(pattern)
        sentences.append(processed_text)
        labels.append(intent['tag'])
        print(f"Pattern: {pattern}, Processed: {processed_text}")

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)
y = np.array(labels)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the vectorizer and model for future use
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def predict_class(text):
    text = preprocess_text(text)
    X_test = vectorizer.transform([text])
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = max(probabilities[0])
    print(f"Processed Text: {text}")
    print(f"Predictions: {predictions}, Confidence: {confidence}")
    if confidence > 0.2:  # Adjust confidence threshold as needed
        return [{'intent': predictions[0], 'probability': confidence}]
    else:
        return [{'intent': 'unknown', 'probability': confidence}]


def query_external_api(query):
    # Example: Querying weather information from an API
    # Replace this with your actual API integration logic
    if 'weather' in query.lower():
        # Example API call
        return "The current weather is 25°C and sunny."
    else:
        return "I'm sorry, I don't have information on that."

def chatbot_response(text):
    intents_list = predict_class(text)
    if intents_list[0]['intent'] == 'unknown':
        # Handle specific questions not covered by intents
        if 'age' in text.lower():
            return get_response([{'intent': 'age'}], intents)
        else:
            return get_response([{'intent': 'fallback'}], intents)  # Fallback intent for other unknown queries
    else:
        return get_response(intents_list, intents)



print("Start chatting with the bot (type 'quit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
