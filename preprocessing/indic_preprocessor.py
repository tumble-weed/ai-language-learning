
"""""
TODO:
Processing incomplete sentences.
"""

import fasttext
import os

INDICLID_MODEL_PATH = "C:\\Users\\OMKAR VIJAY JAGTAP\\Documents\\Language Learning\\indiclid-ftn\\model_baseline_roman.bin"
indiclid_model = fasttext.load_model(INDICLID_MODEL_PATH)

lang_code_map = {
   'asm_Beng': '__label__asm_Beng',
   'asm_Latn': '__label__asm_Latn',
   'ben_Beng': '__label__ben_Beng',
   'ben_Latn': '__label__ben_Latn',
   'brx_Deva': '__label__brx_Deva',
   'brx_Latn': '__label__brx_Latn',
   'doi_Deva': '__label__doi_Deva',
   'doi_Latn': '__label__doi_Latn',
   'eng_Latn': '__label__eng_Latn',
   'guj_Gujr': '__label__guj_Gujr',
   'guj_Latn': '__label__guj_Latn',
   'hin_Deva': '__label__hin_Deva',
   'hin_Latn': '__label__hin_Latn',
   'kan_Knda': '__label__kan_Knda',
   'kan_Latn': '__label__kan_Latn',
   'kas_Arab': '__label__kas_Arab',
   'kas_Deva': '__label__kas_Deva',
   'kas_Latn': '__label__kas_Latn',
   'kok_Deva': '__label__kok_Deva',
   'kok_Latn': '__label__kok_Latn',
   'mai_Deva': '__label__mai_Deva',
   'mai_Latn': '__label__mai_Latn',
   'mal_Mlym': '__label__mal_Mlym',
   'mal_Latn': '__label__mal_Latn',
   'mni_Beng': '__label__mni_Beng',
   'mni_Meti': '__label__mni_Meti',
   'mni_Latn': '__label__mni_Latn',
   'mar_Deva': '__label__mar_Deva',
   'mar_Latn': '__label__mar_Latn',
   'nep_Deva': '__label__nep_Deva',
   'nep_Latn': '__label__nep_Latn',
   'ori_Orya': '__label__ori_Orya',
   'ori_Latn': '__label__ori_Latn',
   'pan_Guru': '__label__pan_Guru',
   'pan_Latn': '__label__pan_Latn',
   'san_Deva': '__label__san_Deva',
   'san_Latn': '__label__san_Latn',
   'sat_Olch': '__label__sat_Olch',
   'snd_Arab': '__label__snd_Arab',
   'snd_Latn': '__label__snd_Latn',
   'tam_Tamil': '__label__tam_Tamil',
   'tam_Latn': '__label__tam_Latn',
   'tel_Telu': '__label__tel_Telu',
   'tel_Latn': '__label__tel_Latn',
   'urd_Arab': '__label__urd_Arab',
   'urd_Latn': '__label__urd_Latn',
   'other': '__label__other'
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



def preprocess_text(text: dict[str, str], language: str) -> list[str]:
   # from text dictionary, filter out sentences not in expected language
   
   # for testing
   # result = {}
   # for k, v in text.items():
   #    if language_identifier(v, language)[0]:
   #       result[k] = v
   #    else:
   #       print(v)

   result = {k: v for k, v in text.items() if language_identifier(v, language)[0]}
   
   return result

if __name__ == "__main__":
   sample_sentence = "माझं नाव ओमकार आहे."

   print(language_identifier(sample_sentence, 'mar_Deva'))
