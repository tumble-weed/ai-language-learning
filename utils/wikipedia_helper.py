import wikipedia
import re

def get_text_from_wikipedia(topic, lang_code="en"):
    """
    Fetches the plain text of a Wikipedia article and uses spaCy
    to segment it into sentences.
    """

    # Set the language for Wikipedia
    wikipedia.set_lang(lang_code)
    
    try:
        # Step 1: Get the text of the Wikipedia article
        # 'content=True' fetches the main content (excluding images, refs, etc.)
        wiki_page = wikipedia.page(topic, auto_suggest=False)
        text = wiki_page.content

        # Step 2: Clean the text
        # Remove headers like "== Section Name =="
        text = re.sub(r'==.*?==\s+', '', text)
        
        # Remove citation superscripts like "[1]"
        text = re.sub(r'\\[.*?\\]', '', text)


    except wikipedia.exceptions.PageError:
        print(f"Error: Wikipedia page for '{topic}' not found.")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Error: Disambiguation needed for '{topic}'. Try a more specific term.")
        print("Possible options:", e.options[:5])
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    return text.replace('\n', ' ').strip()