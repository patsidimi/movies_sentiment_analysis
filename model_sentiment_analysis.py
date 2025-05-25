!pip install --upgrade transformers

!pip install gradio

!pip install wordcloud

import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import nltk
import os
import re
import seaborn as sns
import spacy
import sys
import subprocess
import torch
import base64
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
from google.colab import drive
from wordcloud import WordCloud
from collections import Counter
from graphviz import Digraph

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/Athinorama_movies_dataset.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df.head()

# Υπολογισμός του αριθμού δειγμάτων ανά κατηγορία
samples_per_category = 6000 // 6

print(df.head())
print("\nΠληροφορίες:\n")
print("-" * 50)
print(df.info())

# Έλεγχος για missing values
print("\nΠληροφορίες για missing values:\n")
print("-" * 50)
print(df.isnull().sum())

# Κρατάω χρήσιμες στήλες
df = df[['review', 'stars']]

# Εξάγω τον αριθμό από το κείμενο
df['stars'] = df['stars'].str.extract(r'(\d+)').astype(int)

# Τσεκάρω τα πρώτα δείγματα
df.head()

# Οπτικοποίηση της κατανομής των βαθμολογιών
plt.figure(figsize=(10, 6))
sns.countplot(x='stars', data=df,palette="viridis")
plt.title('Κατανομή Βαθμολογιών Ταινιών')
plt.xlabel('Βαθμολογία')
plt.ylabel('Αριθμός Σχολίων')
plt.show()

#Διαχωρισμός του dataset ανά κατηγορία και τυχαία δειγματοληψία

categories = df['stars'].unique()

# Δημιουργία ενός κενού DataFrame για τα επιλεγμένα δείγματα
balanced_df = pd.DataFrame()

# Για κάθε κατηγορία, επιλέγω τον ίδιο αριθμό δειγμάτων
for category in categories:
    # Φιλτράρισμα του DataFrame για την τρέχουσα κατηγορία
    category_df = df[df['stars'] == category]

    # Τυχαία επιλογή του απαιτούμενου αριθμού δειγμάτων χωρίς επανατοποθέτηση
    category_sample = category_df.sample(n=samples_per_category, random_state=42)

    # Προσθήκη των επιλεγμένων δειγμάτων στο τελικό DataFrame
    balanced_df = pd.concat([balanced_df, category_sample])

# Ανακάτεμα των γραμμών του τελικού dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Επιβεβαίωση της ισορροπίας των κατηγοριών
category_counts = balanced_df['stars'].value_counts()
print("Κατανομή κατηγοριών στο ισορροπημένο dataset:")
print(category_counts)

# Αποθήκευση του νέου ισορροπημένου dataset
balanced_df.to_csv('balanced_dataset.csv', index=False)

# Οπτικοποίηση της κατανομής των βαθμολογιών
plt.figure(figsize=(10, 6))
sns.countplot(x='stars', data=balanced_df,palette="viridis")
plt.title('Κατανομή Βαθμολογιών Ταινιών')
plt.xlabel('Βαθμολογία')
plt.ylabel('Αριθμός Σχολίων')
plt.show()

"""Προκειμένου να αναγνωρίσει καλύτερα αρνήσεις, προσθέτουμε τα παρακάτω δεδομένα για εκπαίδευση"""

negation_data = {
    "review": [
        # Θετικές κριτικές με άρνηση (5 αστέρια)
        "Δεν έχω δει καλύτερη ταινία φέτος.",
        "Δεν υπάρχει καλύτερος ηθοποιός από αυτόν.",
        "Τίποτα δεν συγκρίνεται με αυτό το αριστούργημα.",
        "Καμία ταινία δεν με έχει συγκινήσει περισσότερο.",

        # Θετικές κριτικές με άρνηση (4 αστέρια)
        "Δεν είχα δει τόσο καλή ταινία εδώ και καιρό.",
        "Δεν περίμενα να είναι τόσο διασκεδαστική.",
        "Δεν θα σας απογοητεύσει, είναι εξαιρετική.",

        # Ουδέτερες κριτικές με άρνηση (2.3 αστέρια)
        "Δεν είναι κακή ταινία, αλλά ούτε εντυπωσιακή.",
        "Δεν με απογοήτευσε, αλλά δεν με ενθουσίασε κιόλας.",
        "Δεν είναι τόσο άσχημη όσο περίμενα.",
        "Δεν θα έλεγα ότι είναι χάσιμο χρόνου.",
        "Δεν είναι η καλύτερη δουλειά του σκηνοθέτη, αλλά ούτε η χειρότερη.",
        "Δεν μπορώ να βρω κάτι αρνητικό σε αυτή την ταινία.",

        # Αρνητικές κριτικές με άρνηση (1 αστέρι)
        "Δεν μπορώ να πω καλά λόγια για αυτή την ταινία.",
        "Δεν προσφέρει τίποτα αξιόλογο.",
        "Δεν αξίζει τον χρόνο σας.",

        # Αρνητικές κριτικές με άρνηση (0 αστέρι)
        "Δεν αξίζει τα χρήματά σου, καθόλου καλή.",
        "Δεν μπόρεσα να δω περισσότερο από 20 λεπτά, τόσο κακή.",
        "Δεν πρόκειται να συστήσω αυτή την ταινία σε κανέναν.",
        "Δεν υπάρχει χειρότερη ταινία φέτος."
    ],
    "stars": [
        # Θετικές (5 αστέρια)
        5, 5, 5, 5,

        # Θετικές (4 αστέρια)
        4, 4, 4,

        # Ουδέτερες (2,3 αστέρια)
        3, 3, 3, 2, 2,3,

        # Αρνητικές (1 αστέρια)
        1, 1, 1,

        # Αρνητικές (0 αστέρι)
        0, 0, 0, 0
    ]
}

# Δημιουργία dataframe με τα νέα παραδείγματα
negation_df = pd.DataFrame(negation_data)

# Προβολή των νέων παραδειγμάτων
print(negation_df.head())
print(f"Αριθμός νέων παραδειγμάτων: {len(negation_df)}")

# Συγχώνευση με το αρχικό dataframe
balanced_df_new = pd.concat([balanced_df, negation_df], ignore_index=True)

# Ανακάτεμα των δεδομένων
balanced_df_new = balanced_df_new.sample(frac=1).reset_index(drop=True)
# Προβολή στατιστικών για τις τιμές των stars
print(balanced_df_new['stars'].value_counts().sort_index())

# Αποθήκευση του νέου συνόλου δεδομένων
balanced_df_new.to_csv("balanced_dataset_new.csv", index=False)

def sentiment_category_3(stars):
  if stars <= 1:
    return 0  # Αρνητικό
  elif stars <= 3:
    return 1  # Ουδέτερο
  else:
    return 2  # Θετικό

balanced_df_new['sentiment'] = balanced_df_new['stars'].apply(sentiment_category_3)

balanced_df_new.head()

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=balanced_df_new,palette="viridis")
plt.title('Κατανομή Κριτικών Ταινιών')
plt.xticks([0, 1, 2], ["Αρνητική", "Ουδέτερη", "Θετική"], fontsize=12)
plt.xlabel('Κριτική')
plt.ylabel('Αριθμός Σχολίων')
plt.show()

# Διαχωρισμός σε train και test sets
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df_new['review'],
    balanced_df_new['sentiment'],
    test_size=0.2,
    random_state=42
)

def install_spacy_greek_model():
    print("Εγκατάσταση του ελληνικού μοντέλου spaCy...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "el_core_news_sm"])
        print("Το ελληνικό μοντέλο εγκαταστάθηκε επιτυχώς!")
    except subprocess.CalledProcessError:
        print("Σφάλμα κατά την εγκατάσταση του μοντέλου.")
        return False
    return True

if __name__ == "__main__":
    install_spacy_greek_model()

nltk.download('stopwords')

greek_stopwords = set(stopwords.words('greek'))

print(f"Ελληνικά stopwords: {sorted(greek_stopwords)}")

# Φόρτωση του ελληνικού μοντέλου spaCy

nlp = spacy.load("el_core_news_sm")
def preprocess_text(text, use_spacy=True):
    if not isinstance(text, str):
        return ""

    # Μετατροπή σε πεζά
    text = text.lower()

    # Αφαίρεση URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Αφαίρεση ειδικών χαρακτήρων και αριθμών
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    if use_spacy and nlp:
        # Χρήση spaCy για lemmatization
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc if token.text not in greek_stopwords])
    else:
        # Απλή αφαίρεση stopwords
        text = " ".join([word for word in text.split() if word not in greek_stopwords])

    return text

# Εφαρμογή προεπεξεργασίας
X_train_processed = X_train.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# Δημιουργία TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_processed)
X_test_tfidf = tfidf.transform(X_test_processed)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1, solver='liblinear'),
    "Linear SVM": LinearSVC(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = {}

for name, model in models.items():
    print(f"Εκπαίδευση του μοντέλου {name}...")
    model.fit(X_train_tfidf, y_train)

# Αξιολόγηση
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report
    }

    print(f"{name} - Ακρίβεια: {accuracy:.4f}")
    print(report)
    print("-" * 60)

# Εύρεση του καλύτερου μοντέλου
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"Το καλύτερο μοντέλο είναι: {best_model_name} με ακρίβεια {results[best_model_name]['accuracy']:.4f}")

# Προετοιμασία των δεδομένων για BERT
# Μετατροπή των dataframes σε datasets
train_dataset = Dataset.from_dict({
    'text': X_train.tolist(),
    'label': y_train.tolist()
})

test_dataset = Dataset.from_dict({
    'text': X_test.tolist(),
    'label': y_test.tolist()
})

# Χρησιμοποιώ το Greek BERT
model_name = "nlpaueb/bert-base-greek-uncased-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

#  Προεπεξεργασία με tokenizer

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512  # Περιορισμός μήκους στα 512 tokens
    )

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Ορισμός συνάρτησης υπολογισμού μετρικών
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        'accuracy': accuracy,
        'f1_macro': report['macro avg']['f1-score'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        **{f'f1_{i}': report[str(i)]['f1-score'] for i in range(3)}
    }

#  Εκπαίδευση του μοντέλου
num_labels = len(np.unique(y_train))
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./greek-sentiment-model",
    run_name="greek-sentiment-experiment",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Εκπαίδευση
trainer.train()

# Αξιολόγηση με αναλυτικές μετρικές
results = trainer.evaluate()
print(f"Αποτελέσματα αξιολόγησης: {results}")
# Παραγωγή και εκτύπωση αναλυτικού classification report
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
print(classification_report(y_test, preds, target_names=['Αρνητική', 'Ουδέτερη', 'Θετική']))
# Οπτικοποίηση confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Αρνητική', 'Ουδέτερη', 'Θετική'],
            yticklabels=['Αρνητική', 'Ουδέτερη', 'Θετική'])
plt.xlabel('Προβλεπόμενη Ετικέτα')
plt.ylabel('Πραγματική Ετικέτα')
plt.title('Confusion Matrix για το BERT Μοντέλο')
plt.tight_layout()
plt.show()

# Αποθήκευση του μοντέλου
model.save_pretrained("./greek-sentiment-final-model")
tokenizer.save_pretrained("./greek-sentiment-final-model")

# Δημιουργώ ένα φάκελο στο Drive (αν δεν υπάρχει)
!mkdir -p "/content/drive/My Drive/ML_Models_final"

# Αντιγράφω το φάκελο του μοντέλου στο Drive
!cp -r /content/greek-sentiment-final-model "/content/drive/My Drive/ML_Models_final/"

def predict_sentiment(text, model_type='bert'):
    # Προεπεξεργασία
    processed_text = preprocess_text(text)

    if model_type == 'classic':
        # TF-IDF και πρόβλεψη με το καλύτερο κλασικό μοντέλο
        text_tfidf = tfidf.transform([processed_text])
        prediction = best_model.predict(text_tfidf)[0]

        # Μετατροπή της πρόβλεψης σε ετικέτα
        sentiment_label = ['Αρνητική 😔', 'Ουδέτερη 😐', 'Θετική 😊'][prediction]
        return sentiment_label

    elif model_type == 'bert':
        # Χρήση του BERT μοντέλου
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)
        # Μετακίνηση των input tensors στην ίδια συσκευή με το μοντέλο
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        # Πρόβλεψη
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence = probabilities[0][prediction].item() * 100

    # Μετατροπή της πρόβλεψης σε ετικέτα
    sentiment_labels =  ['Αρνητική 😔', 'Ουδέτερη 😐', 'Θετική 😊']
    result = f"{sentiment_labels[prediction]} (Βεβαιότητα: {confidence:.1f}%)"
    return result

# Παράδειγμα χρήσης
sample_reviews = [
    "Η ταινία ήταν ετσι και ετσι αλλα θα την ξαναεβλεπα",
    "Μέτρια ταινία, είχε κάποιες καλές στιγμές αλλά γενικά ήταν βαρετή.",
    "Τελείως χάλια, χάσιμο χρόνου και χρημάτων."
]

print("Προβλέψεις με το Κλασικό Μοντέλο:")
for review in sample_reviews:
    sentiment = predict_sentiment(review, 'classic')
    print(f"Κείμενο: {review}\nΣυναίσθημα: {sentiment}\n")

print("\nΠροβλέψεις με το BERT Μοντέλο:")
for review in sample_reviews:
    sentiment = predict_sentiment(review, 'bert')
    print(f"Κείμενο: {review}\nΣυναίσθημα: {sentiment}\n")

"""**Gradio**"""

# Φορτώνω τοπική εικόνα σε base64
with open("/content/hidden_figures.jpeg", "rb") as f:
    data = f.read()
data_url = "data:image/jpeg;base64," + base64.b64encode(data).decode()

# Φτιάχνω το HTML για την εικόνα
image_html = f"""
<div style="text-align:center; margin:20px 0;">
  <img src="{data_url}"
       alt="Movie Poster"
       style="max-width:90%; max-height:350px; object-fit:contain; display:inline-block;"
  />
</div>
"""
# Καθαρίζει τις τιμές για το review και το sentiment
def clear_fields():
    return "", ""

#  Blocks για να τοποθετήσω τα components με όποια σειρά θέλω
with gr.Blocks(title="Ανάλυση Συναισθήματος") as demo:
    # Τίτλος
    gr.Markdown(""" # 🎬 Ανάλυση Συναισθήματος Κριτικής Ταινίας """)
    # Περιγραφή
    gr.Markdown("""*Γράψε μια κριτική για μια ταινία και το μοντέλο μας θα αναλύσει το συναίσθημα του κειμένου.*""")
    #  HTML με την κεντραρισμένη εικόνα
    gr.HTML(image_html)

    # Τοποθετώ το textbox και το αποτέλεσμα σε μία σειρά (δίπλα-δίπλα)
    with gr.Row():
        # Αριστερή στήλη: Textbox εισαγωγής
        review = gr.Textbox(
            lines=4,
            placeholder="Γράψε εδώ την κριτική σου για την ταινία...",
            label="Κριτική",

        )

        # Δεξιά στήλη: Πεδίο αποτελέσματος
        sentiment = gr.Textbox(
            label="Αποτέλεσμα",
            lines=4,
        )


    review.submit(predict_sentiment, inputs=review, outputs=sentiment)

    # Δημιουργία row για τα κουμπιά ώστε να εμφανίζονται δίπλα-δίπλα
    with gr.Row():
        # Κουμπί Ανάλυση
        analyze_btn = gr.Button("Ανάλυση 🔍", variant="primary")

        # Κουμπί Καθαρισμός
        clear_btn = gr.Button("Καθαρισμός 🗑️", variant="secondary")

    # Συνδέω τις λειτουργίες με τα κουμπιά
    analyze_btn.click(predict_sentiment, inputs=review, outputs=sentiment)
    clear_btn.click(clear_fields, inputs=None, outputs=[review, sentiment])

demo.launch(share=True)

"""**ΟΠΤΙΚΟΠΟΙΗΣΕΙΣ**

**Δημιουργία FlowChart**


dot = Digraph(comment='flowchart')

# Στάδια
dot.node('A', 'Εισαγωγή Βιβλιοθηκών & Mount Google Drive', shape='box', style='filled', fillcolor='#E0F7FA')
dot.node('B', 'Φόρτωση Dataset', shape='box', style='filled', fillcolor='#FFECB3')
dot.node('C', 'Καθαρισμός Δεδομένων', shape='box', style='filled', fillcolor='#FFE0B2')
dot.node('D', 'Δειγματοληψία για ίσο αριθμό ανά κατηγορία', shape='box', style='filled', fillcolor='#C8E6C9')
dot.node('E', 'Προσθήκη Κριτικών με Άρνηση (Negation)', shape='box', style='filled', fillcolor='#D1C4E9')
dot.node('F', 'Μετατροπή "stars" σε Τρεις Κατηγορίες Sentiment', shape='box', style='filled', fillcolor='#B3E5FC')
dot.node('G', 'Προεπεξεργασία Κειμένου', shape='box', style='filled', fillcolor='#FFCDD2')
dot.node('H', 'TF-IDF Vectorization', shape='box', style='filled', fillcolor='#F0F4C3')
dot.node('I', 'Εκπαίδευση Κλασικών Μοντέλων ML', shape='box', style='filled', fillcolor='#E0F7FA')
dot.node('J', 'Αξιολόγηση Μοντέλων & Επιλογή Καλύτερου', shape='box', style='filled', fillcolor='#FFECB3')
dot.node('K', 'Fine-tuning Greek BERT ', shape='box', style='filled', fillcolor='#FFE0B2')
dot.node('L', 'Αξιολόγηση BERT', shape='box', style='filled', fillcolor='#C8E6C9')

# Σύνδεση των κόμβων
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH','HI','IJ','JK','KL'])

# Αποθήκευση σε αρχείο PNG
dot.render('project_based_learning_flowchart', format='png', cleanup=True)

# Εμφάνιση σε Jupyter Notebook (αν χρειάζεται)
dot.view()

"""**Δημιουργία wordcloud**"""

# Function to clean text for word cloud
def clean_for_wordcloud(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, special characters, numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Remove short words
    words = [word for word in text.split() if len(word) > 2]

    # Remove stopwords
    words = [word for word in words if word not in greek_stopwords]

    return " ".join(words)

# Group texts by sentiment
sentiment_texts = {
    0: [],  # Negative
    1: [],  # Neutral
    2: []   # Positive
}

# Collect all texts by sentiment category
for idx, row in balanced_df_new.iterrows():
    sentiment_texts[row['sentiment']].append(row['review'])

# Clean texts and get word counts for each sentiment
cleaned_texts = {}
word_counts_by_sentiment = {}

for sentiment, texts in sentiment_texts.items():
    # Clean each text
    cleaned = [clean_for_wordcloud(text) for text in texts]
    cleaned_texts[sentiment] = " ".join(cleaned)

    # Count words
    word_counts_by_sentiment[sentiment] = Counter(cleaned_texts[sentiment].split())
    print(f"Total unique words for sentiment {sentiment}: {len(word_counts_by_sentiment[sentiment])}")

# Find common words across all sentiment categories
# Get sets of words from each sentiment that appear more than a threshold
threshold = 5  # Minimum count to consider a word
word_sets = {}
for sentiment, counter in word_counts_by_sentiment.items():
    word_sets[sentiment] = {word for word, count in counter.items() if count > threshold}

# Find words that appear in all sentiment categories
common_words = set.intersection(*word_sets.values())
print(f"\nFound {len(common_words)} common words across all sentiment categories")
print(f"Sample common words: {list(common_words)[:20]}")

# Remove common words from each sentiment's text
filtered_texts = {}
for sentiment, text in cleaned_texts.items():
    words = text.split()
    filtered_words = [word for word in words if word not in common_words]
    filtered_texts[sentiment] = " ".join(filtered_words)

    # Count top words after filtering
    word_counts = Counter(filtered_words)
    print(f"\nTop 15 distinctive words for sentiment {sentiment}: {word_counts.most_common(15)}")

# Create a figure with subplots for each sentiment
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
sentiment_names = {0: 'Αρνητικό 😔', 1: 'Ουδέτερο 😐', 2: 'Θετικό 😊'}

# Create word cloud for each sentiment with filtered texts
for sentiment, text in filtered_texts.items():
    if not text.strip():  # Check if text is empty after filtering
        print(f"Warning: No words left for sentiment {sentiment} after filtering!")
        continue

    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)

    # Plot
    ax = axes[sentiment]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Distinctive Words: {sentiment_names[sentiment]}', fontsize=16)
    ax.axis('off')

plt.tight_layout()
plt.savefig('wordcloud_distinctive.png', dpi=300)
plt.show()

# Also generate standard word clouds without common word filtering for comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for sentiment, text in cleaned_texts.items():
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)

    ax = axes[sentiment]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'All Words: {sentiment_names[sentiment]}', fontsize=16)
    ax.axis('off')

plt.tight_layout()
plt.savefig('wordcloud_all.png', dpi=300)
plt.show()
