from metrics.base import MetricBase
from utils.devnagari_helper import sentence_cleaner
class WordFrequencyMetric(MetricBase):
    def name(self):
        return "WFR"

    def get_frequencies(self, sentences: list[str]):
        freq = {}
        for sentence in sentences:
            clean_sentence = sentence_cleaner(sentence)
            words = clean_sentence.strip().split()
            for word in words:
                freq[word] = freq.get(word, 0) + 1
        return freq

    def compute(self, sentence: str, frequencies: dict = None):
        if frequencies is None:
            print("Frequencies dictionary not provided.")
            return None
        
        clean_sentence = sentence_cleaner(sentence)
        words = clean_sentence.strip().split()
        
        # Calculate relative frequency of each word in the sentence
        total_words = sum(frequencies.values())
        if total_words == 0:
            return 0

        word_frequencies = {word: frequencies.get(word, 0) for word in words}
        relative_frequencies = {word: freq / total_words for word, freq in word_frequencies.items()}
        # print("Relative Frequencies:", relative_frequencies)

        # Return Average Relative Frequency
        avg_relative_frequency = sum(relative_frequencies.values()) / len(words) if words else 0
        return avg_relative_frequency

if __name__ == "__main__":
    # Example usage
    sentences = [
        "यह एक उदाहरण वाक्य है।",
        "यह दूसरा उदाहरण वाक्य है।",
        "यह एक और उदाहरण है।"
    ]

    wfr = WordFrequencyMetric()
    frequencies = wfr.get_frequencies(sentences)
    print("Word Frequencies:", frequencies)

    test_sentence = "यह एक उदाहरण वाक्य है।"
    score = wfr.compute(test_sentence, frequencies)
    print(f"Average Relative Frequency for '{test_sentence}': {score}")