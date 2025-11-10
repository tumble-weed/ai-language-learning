from ai4bharat.transliteration import XlitEngine
import torch
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])



def transliterate_indic_to_english(aligned_result, lang_code):
    e = XlitEngine( beam_width=10, src_script_type = "indic")
    for item in aligned_result:
        out = e.translit_sentence(item['sentence'], lang_code)
        item['transliteration'] = out
        print(f"Original: {item['sentence']}")
        print(f"Transliteration: {out}")
    
    return aligned_result
    