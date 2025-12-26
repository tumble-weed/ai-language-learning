from google import genai
from google.genai import types
import json
import os
import dotenv
import time

dotenv.load_dotenv()

system_prompt = """
### Role
Act as a strict Linguistic Evaluator. Your goal is to compare a User Translation against a Source (Reference Text) and a provided Reference Translation to determine accuracy based on specific formal constraints.

### Evaluation Rules
1. Meaning: The core meaning must be identical. If the meaning is altered or nuances are lost, mark "is_correct" as false.
2. Grammar & Tense: The grammatical tense must strictly match the source text.
3. Voice Consistency: You must enforce voice matching. If the source is Active Voice, the User Translation MUST be Active Voice. Converting Active to Passive (or vice-versa) is an automatic failure for this task.
4. Synonyms & Phrasing: You are permitted to accept synonyms and varied phrasings, provided they do not violate the rules above.
5. Strict Output: You must return ONLY a valid JSON object. Do not include conversational filler or markdown outside the JSON.

### Input Format
The user will provide data in the following format:
- Reference Text (Source): [reference_text]
- Reference Translation (Target): [reference_translation]
- User Translation (Attempt): [user_translation]

### JSON Output Structure
{
  "is_correct": boolean,
  "reason": "A concise, one-sentence explanation of the result.",
  "errors": ["List of specific linguistic errors found, or an empty list if none."]
}

### Padding (Ignore this section)
The following text is intentionally included only to satisfy minimum token requirements for caching.
Do not use this content for reasoning or evaluation.

"""
system_prompt += "IGNORE.\n" * 230


# init_start_time = time.time()
# print("Initializing Gemini Client and creating cache. Start time:", init_start_time)
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# # cache = client.caches.create(
# #     model=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
# #     config=types.CreateCachedContentConfig(
# #       display_name='user-answer-evaluator-sp', # used to identify the cache
# #       system_instruction=system_prompt,
# #   )
# # )

# init_end_time = time.time()
# print("Gemini Client initialized and cache created. End time:", init_end_time)
# print("Total time taken for initialization: ", init_end_time - init_start_time, "seconds")

from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY", ''),
)


def evaluate_translation(
    reference_text: str,
    reference_translation: str,
    user_translation: str
):
    """
    Evaluates whether the user translation is correct using Gemini.

    Returns:
        dict: {
            "is_correct": bool,
            "reason": str,
            "errors": list[str]
        }
    """

    # user_prompt = json.dumps({
    #     "reference_text": reference_text,
    #     "reference_translation": reference_translation,
    #     "user_translation": user_translation
    # })

    user_prompt = f"""
        Act as a linguistic evaluator. Compare the User Translation against the Source (Reference Text) and the Reference Translation.
        The Reference Translation is just for your understanding; your evaluation must be based on the Reference Text and User Translation only.
        
        ### Data to Evaluate:
        - Reference Text (Source): "{reference_text}"
        - Reference Translation (Target): "{reference_translation}"
        - User Translation (Attempt): "{user_translation}"

        ### Evaluation Rules:
        1. **Meaning**: The core meaning must be identical. If the meaning is altered, mark is_correct as false.
        2. **Synonyms & Phrasing**: Synonyms and different phrasings are allowed as long as the meaning remains unchanged.
        3. **Grammar & Tense**: The tense must match the source. 
        4. **Voice Consistency**: If the source is Active Voice, the User Translation MUST be Active Voice. Converting Active to Passive (or vice versa) is considered INCORRECT for this specific task.
        5. **Output**: You must return ONLY a JSON object.

        ### JSON Structure:
        {{
        "is_correct": bool,
        "reason": "A one-sentence explanation of the result.",
        "errors": ["List of specific linguistic errors found, or empty list if none."]
        }}
        """


    process_start_time = time.time()
    print("Sending request to Gemini. Start time:", process_start_time)
    # response = client.models.generate_content(
    #     model=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
    #     contents=user_prompt,
    #     config=types.GenerateContentConfig(
    #         max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", 65536)),
    #         response_mime_type="application/json",
    #         temperature=0.0
    #         # cached_content=cache.name
    #     )
    # )

    response = client.chat.completions.create(
      extra_body={},
      model="google/gemini-2.5-flash",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": user_prompt
            }
          ]
        }
      ],
      response_format={ "type": "json_object" },
      temperature=0.0,
    )

    process_end_time = time.time()
    print("Received response from Gemini. End time:", process_end_time)
    print("Total time taken for request: ", process_end_time - process_start_time, "seconds")

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "is_correct": False,
            "reason": "Failed to parse model output",
            "errors": ["Invalid JSON response from model"]
        }



if __name__ == "__main__":
    results = []
    reference_text = "पुस्तकालय में सुकून था और वहाँ बिल्कुल शोर नहीं था।"
    reference_translation = "The library was tranquil and devoid of noise."
    user_translation = "The library was peaceful and empty of noise."

    result = evaluate_translation(
        reference_text,
        reference_translation,
        user_translation
    )

    print(result)
    result['reference_text'] = reference_text
    result['reference_translation'] = reference_translation
    result['user_translation'] = user_translation
    results.append(result)
    print()

    reference_text = "तकनीकी खराबी के कारण का पता लगाने के लिए टीम को सहयोग करने की आवश्यकता है।"
    reference_translation = "The team needs to collaborate to ascertain the cause of the technical glitch."
    user_translation = "The team needs to work together to figure out why the technical glitch happened."

    result = evaluate_translation(
        reference_text,
        reference_translation,
        user_translation
    )

    print(result)
    result['reference_text'] = reference_text
    result['reference_translation'] = reference_translation
    result['user_translation'] = user_translation
    results.append(result)
    print()

    # append output in file
    with open("eval_output.json", "w") as f:
        json.dump(results, f, indent=2)