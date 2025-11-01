from metrics.base import MetricBase
from utils.devnagari_helper import sentence_cleaner

class SentenceLengthMetric(MetricBase):
    def name(self):
        return "SL"

    def compute(self, sentence: str):
        clean_sentence = sentence_cleaner(sentence)
        words = clean_sentence.strip().split()
        
        length = len(words)
        # print("Sentence Length:", length)
        return length

if __name__ == "__main__":
    # Example usage
    sentences = [
        "यह एक उदाहरण वाक्य है।",
        "यह दूसरा उदाहरण वाक्य है।",
        "       है।"
    ]

    slm = SentenceLengthMetric()
    for sentence in sentences:
        length = slm.compute(sentence)
        print(f"Sentence: {sentence} | Length: {length}")





