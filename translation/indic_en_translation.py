"""
This module provides functionality to translate text from various Indic languages
to English using the AI4Bharat IndicTrans2 model from Hugging Face Transformers.

***
IMPORTANT: This model has specific dependencies.
Please install them using pip:

pip install torch transformers sentencepiece accelerate "git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface"

# For better performance on NVIDIA GPUs, install flash-attention:
pip install flash-attn

The "git+" command installs the 'IndicTransToolkit' which is required for
pre-processing and post-processing the text.
***
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file


HF_TOKEN = os.getenv("HF_TOKEN")

# This import is now possible because of the git+ install command above
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    print("="*80)
    print("ERROR: 'IndicTransToolkit' not found.")
    print('Please install the required dependencies:')
    print('pip install torch transformers sentencepiece accelerate "git+https://github.com/AI4Bharat/IndicTrans2.git#subdirectory=huggingface_interface"')
    print("="*80)
    IndicProcessor = None

# --- Module-level initializations ---

# 1. Define the model checkpoint.
# This is the AI4Bharat model fine-tuned for Indic-to-English translation.
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
# MODEL_NAME = "prajdabre/rotary-indictrans2-indic-en-1B"

# 2. Set up device (GPU if available, else CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"IndicTrans2 Translator: Using device: {DEVICE}")

# 3. Load the tokenizer, model, and processor ONCE when the module is imported.
try:
    if IndicProcessor:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)

        # --- Updated Model Loading ---
        # Set up model arguments based on device
        model_load_args = {
            "trust_remote_code": True,
        }

        if DEVICE.type == "cuda":
            print("IndicTrans2 Translator: Loading model with CUDA optimizations (float16 & Flash Attention 2)")
            model_load_args["torch_dtype"] = torch.float16
            try:
                # Try to use Flash Attention 2 if available
                model_load_args["attn_implementation"] = "flash_attention_2"
            except ImportError:
                print("WARNING: flash-attn not installed. Falling back to default attention.")
                print("Run 'pip install flash-attn' for faster inference.")
        elif DEVICE.type == "mps":
            print("IndicTrans2 Translator: Loading model with MPS optimizations (float16)")
            model_load_args["torch_dtype"] = torch.float16
        else:
            print("IndicTrans2 Translator: Loading model on CPU (float32).")
            model_load_args["low_cpu_mem_usage"] = True # Keep this for CPU

        MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            **model_load_args
        ).to(DEVICE)
        # --- End of Updated Model Loading ---

        MODEL.eval()  # Set model to evaluation mode

        # Initialize the processor for pre/post-processing (as per user's sample)
        IP = IndicProcessor(inference=True)

        print(f"IndicTrans2 Translator: Model '{MODEL_NAME}' loaded successfully.")
    else:
        TOKENIZER, MODEL, IP = None, None, None
        
except Exception as e:
    print(f"Error loading IndicTrans2 model or tokenizer: {e}")
    print("Please ensure you have run the correct pip install command (see docstring).")
    TOKENIZER, MODEL, IP = None, None, None

# 4. Language code mapping
# IndicTrans2 uses 3-letter ISO 639-3 codes with script identifiers.
INDIC_LANG_CODE_TO_IT2_CODE = {
    "hi": "hin_Deva",  # Hindi
    "mr": "mar_Deva",  # Marathi
    "bn": "ben_Beng",  # Bengali
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "gu": "guj_Gujr",  # Gujarati
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "pa": "pan_Guru",  # Punjabi
    "ur": "urd_Arab",  # Urdu
    "ne": "npi_Deva",  # Nepali (npi)
    "si": "sin_Sinh",  # Sinhala
    "as": "asm_Beng",  # Assamese
    "or": "ory_Orya",  # Oriya
    # --- Add all 22 scheduled languages ---
    "brx": "brx_Deva", # Bodo
    "doi": "doi_Deva", # Dogri
    "gom": "gom_Deva", # Konkani
    "kas": "kas_Arab", # Kashmiri (Arabic script)
    "mai": "mai_Deva", # Maithili
    "mni": "mni_Beng", # Manipuri (Bengali script)
    "sat": "sat_Olck", # Santali
    "sd": "snd_Arab",  # Sindhi (Arabic script)
    "sa": "san_Deva",  # Sanskrit
}
TARGET_LANG = "eng_Latn" # English

def translate_indic_to_english(aligned_result: List[Dict], lang_code: str) -> List[Dict]:
    """
    Translates a list of sentences from a specified Indic language to English
    using the AI4Bharat IndicTrans2 model.

    The function updates the input list of dictionaries in-place by adding
    a 'translation' key to each dictionary.

    Args:
        aligned_result: A list of dictionaries, where each dict has at least
                        a 'sentence' key with the text to translate.
        lang_code: The 2-letter (or 3-letter) ISO language code for the
                   source language (e.g., 'hi', 'mr', 'kas').

    Returns:
        The same list of dictionaries, now mutated to include a 'translation' key.

    Raises:
        ValueError: If the model failed to load or the lang_code is not supported.
    """
    if not MODEL or not TOKENIZER or not IP:
        raise ValueError(
            "IndicTrans2 model, tokenizer, or processor not loaded. "
            "Check for errors on import and ensure dependencies are installed."
        )

    # 1. Map the language code to the IndicTrans2 code (e.g., "mr" -> "mar_Deva")
    if lang_code not in INDIC_LANG_CODE_TO_IT2_CODE:
        raise ValueError(
            f"Unsupported language code: '{lang_code}'. "
            f"Supported codes are: {list(INDIC_LANG_CODE_TO_IT2_CODE.keys())}"
        )
    source_language_code = INDIC_LANG_CODE_TO_IT2_CODE[lang_code]
    
    # Handle empty input list
    if not aligned_result:
        return []

    # 2. Get all sentences from the list
    sentences_to_translate = [item['sentence'] for item in aligned_result]

    print(f"Translating {len(sentences_to_translate)} sentences from {source_language_code} to {TARGET_LANG}...")

    # 3. Pre-process the batch using IndicProcessor
    # This adds the special source/target language tokens.
    batch = IP.preprocess_batch(
        sentences_to_translate,
        src_lang=source_language_code,
        tgt_lang=TARGET_LANG
    )

    # 4. Batch Tokenization
    inputs = TOKENIZER(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256 # IndicTrans2 default is 256
    ).to(DEVICE)

    # 5. Batch Generation (Inference)
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True,
            # --- Added from user's sample code ---
            use_cache=True,
            min_length=0,
            num_return_sequences=1
        )

    # 6. Batch Decoding (from IDs back to text)
    decoded_translations = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)

    # 7. Post-process the batch using IndicProcessor
    # This cleans up any remaining special tokens or artifacts.
    translations = IP.postprocess_batch(decoded_translations, lang=TARGET_LANG)

    # 8. Update the original list of dictionaries
    for i, item in enumerate(aligned_result):
        item['translation'] = translations[i]
    
    print("Translation complete.")
    return aligned_result

# --- Example Usage ---
# This block will only run when you execute the script directly
if __name__ == "__main__":
    # telugu sentences
    test_data = [
        {"sentence": "హ నన్ను కనిపెట్టలేర్లే లైట్ వాళ్ళ ఇంటికి ఆ చీఫ్ వస్తాడు లైట్ ఇలా అంటాడు ఏంటి నాన్న ఇవాళ వర్క్ త్వరగా అయిపోయిందా అని ఎస్ లైట్ వాళ్ళ ఫాదర్ షిఫె ఇంకా లైట్ వాళ్ళ ఫాదర్ లైట్ తో తన స్టడీస్ గురించి డిస్కస్ చేయడం స్టార్ట్ చేస్తాడు ఏంటి లైట్ బాగానే చదువుతున్నావా చదువు మీద కాన్సన్ట్రేట్ చేస్తున్నావా అని లైట్ కూడా ఆ నాన్న బానే చదువుతున్నాను అని అంటాడు తన సిస్టర్ కూడా ఆ అవును అన్న తను బానే చదువుతాడు నాకు బానే ఎక్స్ప్లెయిన్ చేస్తాడు కాని అంటుంది వాళ్ళ ఫాదర్ కొంచెం స్ట్రెస్లో ఉంటాడు అదే కేసు తేలట్లేదండి లైట్ వాళ్ళ ఫాదర్ ఏ సిస్టంలో అయితే ఇన్వెస్టిగేషన్కి సంబంధించిన ఇన్ఫర్మేషన్ని స్టోర్ చేశాడో లైట్ అదే సిస్టమ్ని హ్యాక్ చేశాడు లైట్ తన ఫాదర్ సిస్టమ్ని చాలా ఈజీగా హ్యాక్ చేస్తాడు తనకి అది ముందే తెలుసు ఈరోజు ఇన్వెస్టిగేషన్ రిపోర్ట్ ఇచ్చినప్పుడు ఎల్ కి వాళ్ళు ఏ ఇన్ఫర్మేషన్ అయితే చెప్పారో లైట్ ఆ ఇన్ఫర్మేషన్ తెలుసుకున్నాడు వాళ్ళు కీరా ఒక స్టూడెంట్ అనుకుంటున్నారు అనే ఇన్ఫర్మేషన్ ఇప్పుడు లైట్కి తెలుసు లైట్ ఇలా అనుకుంటాడు ఒకవేళ నేను ఇప్పుడు డెత్ నోట్లు ఎవరిని ఏమైనా రాస్తే నెక్స్ట్ ఫార్టీ సెకండ్స్ లోపు వాళ్ళు ఎలా చనిపోవాలో రాయాలి అలా రాయకుంటే గనక వాళ్ళు హార్ట్అటాక్ వచ్చి చనిపోతారు అండ్ వాళ్ళు చనిపోయిన సిక్స్ మినిట్స్ తర్వాత వాళ్ళు ఎలా చనిపోయారు అనే విషయం నేను రాయాలి అదే ఒకవేళ నేను వాళ్ళు ఎలా చనిపోవాలో కూడా రాస్తే వాళ్ళు చనిపోయిన సిక్స్ మినిట్స్ తర్వాత నేను ఆ విషయం రాయాల్సిన అవసరం లేదు"}
    ]

    translated_data = translate_indic_to_english(test_data, lang_code="te")