import webview
import re
from nltk.corpus import reuters
from spellchecker import SpellChecker


class Api:
    def __init__(self):
        corpus = reuters.sents(categories=reuters.categories())
        self.spellchecker = SpellChecker('vocab.txt', corpus, 'count_1edit.txt', max_distance=1, ngram=2)

    def process_text(self, text):
        corrected_text = self.spellchecker.process_line(text)
        # 删除标点符号前的空格
        corrected_text = re.sub(r'\s+([,.!?;:])', r'\1', corrected_text)
        # 计算差异以进行高亮显示
        differences = self.calculate_differences(text, corrected_text)
        return {"corrected_text": corrected_text, "differences": differences}

    def calculate_differences(self, original, corrected):
        # Split text into words
        # 打印出来看看
        original_words = original.split()
        corrected_words = corrected.split()

        differences = []

        # Compare word by word
        max_len = max(len(original_words), len(corrected_words))
        for i in range(max_len):
            if i < len(original_words) and i < len(corrected_words):
                if original_words[i] != corrected_words[i]:
                    differences.append(i)
            elif i < len(original_words):
                differences.append(i)
            elif i < len(corrected_words):
                differences.append(i)

        return differences


def start_app():
    api = Api()
    window = webview.create_window('Spell Checker', 'static/index.html', js_api=api)
    webview.start()

if __name__ == '__main__':
    start_app()
