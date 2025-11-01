from metrics.base import MetricBase
from utils.devnagari_helper import is_devanagari_consonant

class JodaAksharMetric(MetricBase):

  def name(self) -> str:
    """Unique Name for the metric"""
    return "jodakshar_count"

  def compute(self, sentence: str) -> float:
    """
    Counts the number of Joda Akshar (conjunct consonants) in a Devanagari sentence.

    A Joda Akshar is identified by the base pattern:
    Consonant1 + Virama (Halant, U+094D) + Consonant2
    """
    count = 0
    virama = '\u094D'

    i = 0
    while i < len(sentence):
        char = sentence[i]

        # Check if current char is a consonant
        if is_devanagari_consonant(char):
            # If it's a consonant, look ahead for the Virama + Consonant pattern
            if i + 2 < len(sentence):
                next_char = sentence[i+1]
                next_next_char = sentence[i+2]

                # Check for the C1 + V + C2 pattern
                if next_char == virama and is_devanagari_consonant(next_next_char):
                    # Found the start of a Joda Akshar
                    count += 1
                    i = i + 2

                    while i + 2 < len(sentence) and sentence[i+1] == virama and is_devanagari_consonant(sentence[i+2]):
                        i += 2
        i += 1
    return float(count)