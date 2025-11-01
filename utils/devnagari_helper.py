import string
DEVANAGARI_PUNCTUATION = "।॥"
PUNCTUATION_TO_REMOVE = string.punctuation + DEVANAGARI_PUNCTUATION + string.digits

translator = str.maketrans('', '', PUNCTUATION_TO_REMOVE)

def is_devanagari_consonant(char):
    """
    Checks if a character is a Devanagari consonant.
    """
    # Main range U+0915 (क) to U+0939 (ह)
    if '\u0915' <= char <= '\u0939':
        return True

    # Extended range U+0958 (क़) to U+095F (य़)
    if '\u0958' <= char <= '\u095F':
        return True

    return False

def is_devanagari_vowel(char):
    """Checks if a character is an independent Devanagari vowel."""
    if '\u0905' <= char <= '\u0914':
        return True
    return False

def sentence_cleaner(sentence: str) -> str:
    """
    Cleans a Devanagari sentence by removing punctuation and digits.
    """

    clean_sentence = sentence.translate(translator)
    return clean_sentence
