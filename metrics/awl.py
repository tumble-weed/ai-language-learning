from metrics.base import MetricBase
import regex
from utils.devnagari_helper import sentence_cleaner


class AvgWordLengthMetric(MetricBase):
    def name(self):
        return "AWL"

    def compute(self, sentence: str):
        
        
        # 1. Create a "translation table" to efficiently remove all punctuation/digits
        clean_sentence = sentence_cleaner(sentence)

        # 2. Split into words. Now "है।" truly becomes "है"
        words = clean_sentence.strip().split()

        # print(f"Cleaned Words: {words}")

        total_units = 0
        for word in words:
            # 3. Use regex.findall(r'\X', ...) to count grapheme clusters (aksharas)
            # \X matches a single Unicode grapheme cluster.
            akshara_len = len(regex.findall(r'\X', word))
            # print(f"Word: {word}, Akshara Length: {akshara_len}")
            total_units += akshara_len

        # 4. Calculate average
        return total_units / len(words) if words else 0

if __name__ == "__main__":
    metric = AvgWordLengthMetric()
    test_sentence = "यह एक परीक्षण वाक्य है।"
    result = metric.compute(test_sentence)
    print(f"Average Word Length for the sentence: {result}")