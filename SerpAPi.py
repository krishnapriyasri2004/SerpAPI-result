import requests
import random
import time
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def get_google_search_results(search_query):
    # Google search URL
    search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"

    # Generate a random User-Agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36",
        # Add more user agents as needed
    ]
    headers = {'User-Agent': random.choice(user_agents)}

    # Add a random delay
    time.sleep(random.uniform(1, 3))  # Random delay between 1 to 3 seconds

    # Send GET request to Google search URL
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find search results
        search_results = soup.find_all('div', class_='tF2Cxc')

        return search_results
    else:
        return None

def extract_details_from_search_results(search_results):
    details = []

    for result in search_results:
        # Extract text from search result
        text = result.get_text(strip=True)
        details.append(text)

    return details

def save_to_text_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(item + '\n')

def generate_word_cloud(data):
    text = ' '.join(data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def perform_content_mining(filename):
    # Read text from file
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenization
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Frequency analysis
    word_freq = Counter(tokens)

    # Named Entity Recognition (NER)
    entities = [(entity.text, entity.label_) for entity in doc.ents]

    return tokens, word_freq, entities

def search_keywords_in_details(details, keywords):
    matched_sentences = []
    for detail in details:
        for keyword in keywords:
            if keyword.lower() in detail.lower():
                matched_sentences.append(detail)
                break  # Break after finding the first match in a sentence
    return matched_sentences

def train_logistic_regression(X_train, y_train):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train_tfidf, y_train)
    return logistic_regression, tfidf_vectorizer

if __name__ == "__main__":
    search_query = input("Enter the name of a celebrity, movie, or web series: ")
    search_results = get_google_search_results(search_query)
    
    if search_results:
        details = extract_details_from_search_results(search_results)
        for detail in details:
            print(detail)
        
        save_to_text_file(details, f"{search_query}_details.txt")
        print("Details saved to file successfully!")
        
        generate_word_cloud(details)

        # Label the data (for demonstration purposes, manually label relevant and irrelevant)
        labels = []  # 0 for irrelevant, 1 for relevant
        for detail in details:
            label = int(input(f"Is the following detail relevant? \n{detail}\n(Enter 1 for Yes, 0 for No): "))
            labels.append(label)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(details, labels, test_size=0.2, random_state=42)

        # Train logistic regression model
        logistic_regression_model, tfidf_vectorizer = train_logistic_regression(X_train, y_train)

        # Predict labels for the test set
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        y_pred = logistic_regression_model.predict(X_test_tfidf)

        # Calculate precision, recall, and F-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-score: {f_score}")
        
        # Analysis
        if precision == 0 and recall == 0:
            print("No relevant details found.")
        elif precision == 0:
            print("No relevant details found, but there were false positives.")
        elif recall == 0:
            print("All relevant details were missed.")
        elif precision == 1 and recall == 1:
            print("All relevant details found with no false positives.")
        else:
            print("A mix of relevant and irrelevant details were found.")
        
        # Perform content mining on the saved file
        tokens, word_freq, entities = perform_content_mining(f"{search_query}_details.txt")
        
        # Print content mining results
        print("\nContent Mining Results:")
        print("Tokens:", tokens[:10])  # Print first 10 tokens
        print("Word Frequencies:", word_freq.most_common(10))  # Print 10 most common words
        print("Named Entities:", entities[:10])  # Print first 10 named entities
        
        # Keyword search
        user_keywords = input("Enter keywords to search within the details (comma-separated): ").split(',')
        user_keywords = [keyword.strip() for keyword in user_keywords]
        matched_sentences = search_keywords_in_details(details, user_keywords)
        if matched_sentences:
            print("\nMatched Sentences:")
            for sentence in matched_sentences:
                print(sentence)
        else:
            print("No matches found for the provided keywords.")
    else:
        print("Error accessing Google search. Please try again later")
