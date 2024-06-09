import webview
from spellchecker import SpellChecker
from nltk.corpus import reuters

class Api:
    def __init__(self):
        corpus = reuters.sents(categories=reuters.categories())
        self.spellchecker = SpellChecker('vocab.txt', corpus, 'count_1edit.txt', max_distance=1, ngram=2)

    def process_text(self, text):
        corrected_text = self.spellchecker.process_line(text)
        return {"corrected_text": corrected_text}

def start_app():
    api = Api()
    window = webview.create_window('Spell Checker', 'static/index.html', js_api=api)
    webview.start()

if __name__ == '__main__':
    start_app()
