from cadence import PunctuationModel

MODEL_DIR = "..\\cadence"

# Load model (local path)
model = PunctuationModel(
    model_path=MODEL_DIR,
    cpu=True,
    max_length=512,  # length for trunation; also used as window size when sliding_window=True
    attn_implementation="eager",
    sliding_window=True,  # Handle long texts
    verbose=False,  # Quiet mode
    d_type="float32",
)

def restore_punctuation(input_text: str) -> list[str]:
    """
    Restore punctuation in the given input text using the Cadence PunctuationModel.

    Args:
        input_text (str): The text to restore punctuation for.
    """
    result = model.punctuate([input_text], batch_size=1)
    return result[0]