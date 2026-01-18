import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download nltk packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def preprocess_text(text):
    # check if text is a string
    if type(text) != str:
        return ""

    # 1. lower case
    text = text.lower()
    
    # 2. remove punctuation
    # preserve some meaning if needed, but standard request was removing it. 
    # for sentiment, sometimes ! or ? is useful but standard cleaning often removes them.
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. remove whitespaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 4. tokenization
    tokens = word_tokenize(text)
    
    # 5. remove stopwords but keep negations for sentiment analysis
    stop_words = stopwords.words('english')
    
    # list of negative words to keep
    negations = ['no', 'not', 'nor', 'neither', 'never', 'none', "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"]
    
    # simple loop to remove negations from the stop_words list
    final_stop_words = []
    for sw in stop_words:
        if sw not in negations:
            final_stop_words.append(sw)
            
    new_tokens = []
    for word in tokens:
        if word not in final_stop_words:
            new_tokens.append(word)
    tokens = new_tokens
    
    # 6. lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = []
    for word in tokens:
        lemma_tokens.append(lemmatizer.lemmatize(word))
    tokens = lemma_tokens
    
    # join back to string
    final_text = " ".join(tokens)
    return final_text

# main part
def main():
    print("Reading dataset...")
    # changing to train.csv as requested for model training
    file_name = 'train.csv'
    try:
        df = pd.read_csv(file_name)
    except:
        print(f"File {file_name} not found!")
        exit()

    # fill missing values
    print("Handling missing values...")
    
    if 'Review Text' in df.columns:
        df['Review Text'] = df['Review Text'].fillna(" ")
    
    if 'Review Title' in df.columns:
        df['Review Title'] = df['Review Title'].fillna(" ")

    # basic cleaning and other steps
    print("Preprocessing Review Text...")
    if 'Review Text' in df.columns:
        df['Review Text'] = df['Review Text'].apply(preprocess_text)
    
    print("Preprocessing Review Title...")
    if 'Review Title' in df.columns:
        df['Review Title'] = df['Review Title'].apply(preprocess_text)
        
    # keep Star Rating column
    if 'Star Rating' not in df.columns:
        print("Warning: Star Rating column missing!")
        
    print("Saving to preprocessed_train_all.csv (all columns)...")
    df.to_csv('preprocessed_train_all.csv', index=False)
    
    print("Saving to preprocessed_train_features.csv (model ready)...")
    columns_to_save = ['Review Text', 'Review Title', 'Star Rating']
    # check if they exist before saving subset
    final_cols = []
    for col in columns_to_save:
        if col in df.columns:
            final_cols.append(col)
    
    if final_cols:
        df[final_cols].to_csv('preprocessed_train_features.csv', index=False)
        
    print("Completed!")

if __name__ == "__main__":
    main()
