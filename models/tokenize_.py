import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.compile(url_regex).findall(text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens