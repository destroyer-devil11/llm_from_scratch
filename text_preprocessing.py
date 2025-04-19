import re
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords  # 🔧 fixed: was capitalized
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


class Preprocessor:
    def __init__(self, text):
        self.text = text.lower()

    def remove_html(self):
        self.text = re.sub(r"<.*?>", "", self.text)
        return self

    def remove_urls(self):
        self.text = re.sub(r"http\S+|www\S+", "", self.text)
        return self

    def remove_punctuation(self):
        self.text = self.text.translate(str.maketrans("", "", string.punctuation))
        return self

    def remove_numbers(self):
        self.text = re.sub(r"\d+", "", self.text)
        return self

    def remove_whitespace(self):
        self.text = " ".join(self.text.split())
        return self

    def remove_non_ascii(self):
        self.text = self.text.encode("ascii", "ignore").decode()
        return self

    def correct_spelling(self):
        spell = SpellChecker()
        self.text = " ".join([spell.correction(word) for word in self.text.split()])
        return self

    def remove_stopwords(self):
        stop_words = set(stopwords.words("english"))
        self.text = " ".join([w for w in self.text.split() if w not in stop_words])
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.text = " ".join(
            [lemmatizer.lemmatize(w) for w in word_tokenize(self.text)]
        )
        return self

    def get_text(self):
        return self.text


from nltk import download

download("punkt")
download("wordnet")
download("stopwords")

text = (
    "<p>This is a test sentence! With some 123 numbers and a URL: https://test.com</p>"
)

cleaned = (
    Preprocessor(text)
    .remove_html()
    .remove_urls()
    .remove_punctuation()
    .remove_numbers()
    .remove_non_ascii()
    .remove_whitespace()
    .remove_stopwords()
    .lemmatize()
    # .correct_spelling()  # optional, slow
    .get_text()
)

print(cleaned)
