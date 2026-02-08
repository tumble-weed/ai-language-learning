
"""""
TODO:
Processing incomplete sentences.
"""

import fasttext
import os

INDICLID_MODEL_PATH = ".\\indiclid-ftn\\model_baseline_roman.bin"
indiclid_model = fasttext.load_model(INDICLID_MODEL_PATH)

lang_code_map = {
   'as': '__label__asm_Beng',
   'bn': '__label__ben_Beng',
   'brx': '__label__brx_Deva',
   'doi': '__label__doi_Deva',
   'gu': '__label__guj_Gujr',
   'hi': '__label__hin_Deva',
   'kn': '__label__kan_Knda',
   'ks': '__label__kas_Arab',
   'kok': '__label__kok_Deva',
   'mai': '__label__mai_Deva',
   'ml': '__label__mal_Mlym',
   'mni': '__label__mni_Meti',
   'mr': '__label__mar_Deva',
   'ne': '__label__nep_Deva',
   'or': '__label__ori_Orya',
   'pa': '__label__pan_Guru',
   'sa': '__label__san_Deva',
   'sat': '__label__sat_Olch',
   'sd': '__label__snd_Arab',
   'ta': '__label__tam_Tamil',
   'te': '__label__tel_Telu',
   'ur': '__label__urd_Arab',
}


def language_identifier(sentence: str, expected_lang_code: str, k: int = 2):
   """
      Identify if the sentence is in the expected language.

      Args:
         sentence (str): The sentence to check.
         expected_lang_code (str): The expected language code.
         k (int): Number of top predictions to consider.

      Returns:
         tuple: (is_expected_language (bool), predicted_label (str), confidence_score (float))
   """
   predictions = indiclid_model.predict(sentence, k=k)
   predicted_labels = predictions[0]

   return (lang_code_map[expected_lang_code] in predicted_labels, predicted_labels, predictions[1])


def remove_duplicates(sentences: list[str]) -> list[str]:
   """
      Remove duplicate sentences from the list.

      Args:
         sentences (list): List of sentences.

      Returns:
         list: List with duplicates removed.
   """
   # unique_sentences = list(set(sentences))
   # return unique_sentences

   
   # Remove duplicates without changing order
   seen = set()
   unique_sentences = []
   for sentence in sentences:
       if sentence not in seen:
           seen.add(sentence)
           unique_sentences.append(sentence)
   return unique_sentences

def split_by_punctuation(text: str) -> list[str]:
   """
      Split text into sentences based on punctuation marks.

      Args:
         text (str): The input text.

      Returns:
         list: List of sentences.
   """
   import re

   # Define punctuation marks for splitting
   punctuation_marks = r'[।.!?]+'

   # Split text using regex
   sentences = re.split(punctuation_marks, text)

   # Remove any leading/trailing whitespace from sentences
   sentences = [sentence.strip() for sentence in sentences]
   return sentences

def preprocess_text(text: str, language: str) -> list[str]:

   # Split text into sentences based on punctuation
   sentences = split_by_punctuation(text)

   # Remove duplicate sentences
   sentences = remove_duplicates(sentences)

   # from text dictionary, filter out sentences not in expected language
   
   # for testing
   # result = []
   # for sentence in sentences:
   #    res = language_identifier(sentence, language)
   #    if res[0]:
   #       result.append(sentence)
   #    else:
   #       print(sentence)

   result = [sentence for sentence in sentences if language_identifier(sentence, language)[0]]
   
   return result

if __name__ == "__main__":
   sample_sentence = "माझं नाव ओमकार आहे."

   print(language_identifier(sample_sentence, 'mar_Deva'))
