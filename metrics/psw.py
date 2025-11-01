from metrics.base import MetricBase
from utils.devnagari_helper import is_devanagari_consonant, is_devanagari_vowel, sentence_cleaner

class PolySyllabicWordMetric(MetricBase):
    def name(self) -> str:
        """Unique Name for the metric"""
        return "polysyllabic_word_count"

    def _count_syllables(self, word: str) -> int:
        """
        Counts the syllables in a single Devanagari word.

        Logic: A syllable is counted for:
        1. Every independent vowel (अ, आ, इ...).
        2. Every consonant (क, ख, ग...) that is NOT followed by a
           Virama (्), as the Virama suppresses its inherent vowel.
        """
        count = 0
        virama = '\u094D'
        i = 0

        while i < len(word):
            char = word[i]

            # 1. Check for independent vowel
            if is_devanagari_vowel(char):
                count += 1

            # 2. Check for consonant
            elif is_devanagari_consonant(char):
                # Look ahead to see if it's followed by Virama
                if (i + 1 < len(word)) and (word[i+1] == virama):
                    # This consonant is part of a conjunct and doesn't
                    # form its own syllable. We'll skip it.
                    pass
                else:
                    # This is a full syllable (e.g., 'क', 'का', 'कि')
                    count += 1

            # Move to the next character
            i += 1

        return count

    def compute(self, sentence: str) -> float:
        """
        Compute the metric for a given sentence.
        Counts words with more than 2 syllables.
        """
        psw_count = 0

        # Remove common punctuation before splitting
        # This regex removes ., ,, !, ?, |
        cleaned_sentence = sentence_cleaner(sentence)

        # Split the sentence into words
        words = cleaned_sentence.split()

        for word in words:
            syllable_count = self._count_syllables(word)

            # Definition: "count of syllable exceeds 2" (i.e., > 2)
            if syllable_count > 2:
                psw_count += 1

        return float(psw_count)