import spacy
from typing import Dict, List, Optional
import pandas as pd

# Load English Model
nlp = spacy.load("en_core_web_md")

CLAUSE_DEPS = {
    "advcl",       # adverbial clause
    "ccomp",       # finite clausal complement
    "xcomp",       # non-finite clausal complement
    "acl",         # clausal modifier of noun (includes many "modifier clauses")
    "relcl",       # relative clause
    "acl:relcl",   # sometimes appears in UD-aligned models
    "csubj",       # clausal subject
    "csubjpass",   # clausal subject (passive)
    "parataxis"    # non-subordinate clause, but clause-level complexity
}

def extract_tense_features_from_doc(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    """
    Analyzes a spaCy Doc object to create 12 binary features representing the 
    standard English tenses.
    
    Args:
        doc: A spaCy Doc object containing the parsed sentence
        
    Returns:
        Dictionary with 12 binary tense features (0 or 1 for each tense)
    """
    
    # Initialize the 12 Tense Feature Vector (All Zeros)
    tense_features = {
        'Tense_Simple_Present': 0, 
        'Tense_Present_Continuous': 0, 
        'Tense_Present_Perfect': 0, 
        'Tense_Present_Perfect_Cont': 0,
        'Tense_Simple_Past': 0, 
        'Tense_Past_Continuous': 0, 
        'Tense_Past_Perfect': 0, 
        'Tense_Past_Perfect_Cont': 0,
        'Tense_Simple_Future': 0, 
        'Tense_Future_Continuous': 0, 
        'Tense_Future_Perfect': 0, 
        'Tense_Future_Perfect_Cont': 0,
    }
    
    # Find the main verb (root) and collect all auxiliary verbs
    root_verb: Optional[spacy.tokens.Token] = None
    aux_chain: List[spacy.tokens.Token] = []
    main_verbs: List[spacy.tokens.Token] = []
    
    # Single pass through document to collect relevant tokens
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            root_verb = token
        if token.pos_ == "AUX":
            aux_chain.append(token)
        if token.pos_ == "VERB":
            main_verbs.append(token)
    
    # If no clear root, try to find main verb
    if root_verb is None and main_verbs:
        root_verb = main_verbs[0]
    
    # If still no verb found, return all zeros
    if root_verb is None:
        return tense_features
    
    # Core Flags
    is_present = False
    is_past = False
    is_future = False
    is_perfect = False
    is_continuous = False
    
    # Analyze auxiliary chain
    has_will_shall = False
    has_have = False
    has_be_continuous = False
    has_be_form = False
    
    for aux in aux_chain:
        lemma = aux.lemma_.lower()
        
        # Future markers: will, shall, going to (be going to)
        if lemma in ('will', 'shall'):
            has_will_shall = True
            is_future = True
        
        # Perfect aspect: have/has/had + past participle
        if lemma == 'have':
            has_have = True
            is_perfect = True
            # Determine time from 'have' form
            if aux.tag_ in ('VBP', 'VBZ'):  # have/has
                is_present = True
            elif aux.tag_ == 'VBD':  # had
                is_past = True
        
        # Continuous aspect and time determination from 'be'
        if lemma == 'be':
            has_be_form = True
            # Determine time from 'be' form if not already set by 'have'
            if not has_have and not has_will_shall:
                if aux.tag_ in ('VBP', 'VBZ', 'VB'):  # am/is/are
                    is_present = True
                elif aux.tag_ == 'VBD':  # was/were
                    is_past = True
    
    # Check for continuous: need 'be' + VBG (present participle)
    # Look for VBG after any form of 'be' in the sentence
    if has_be_form:
        for i, token in enumerate(doc):
            if token.pos_ == "AUX" and token.lemma_.lower() == 'be':
                # Check if there's a VBG after this 'be' (within reasonable distance)
                for j in range(i + 1, min(i + 4, len(doc))):
                    if doc[j].tag_ == 'VBG' and doc[j].pos_ == 'VERB':
                        is_continuous = True
                        break
                if is_continuous:
                    break
    
    # Check for "be going to" future construction
    if not is_future:
        for i, token in enumerate(doc):
            if (token.lemma_.lower() == 'be' and 
                i + 2 < len(doc) and 
                doc[i + 1].lower_ == 'going' and 
                doc[i + 2].lower_ == 'to'):
                is_future = True
                break
    
    # Analyze main verb if no auxiliaries clearly indicate time
    if not (is_present or is_past or is_future) and root_verb:
        if root_verb.morph.get("Tense") == ["Pres"]:
            is_present = True
        elif root_verb.morph.get("Tense") == ["Past"]:
            is_past = True
        # Fallback to tag-based detection
        elif root_verb.tag_ == 'VBD':
            is_past = True
        elif root_verb.tag_ in ('VBP', 'VBZ', 'VB'):
            is_present = True
    
    # Determine Final Tense/Aspect Combination
    # Order matters: check compound tenses before simple tenses
    
    if is_future:
        if is_perfect and is_continuous:
            tense_features['Tense_Future_Perfect_Cont'] = 1
        elif is_perfect:
            tense_features['Tense_Future_Perfect'] = 1
        elif is_continuous:
            tense_features['Tense_Future_Continuous'] = 1
        else:
            tense_features['Tense_Simple_Future'] = 1
    
    elif is_present:
        if is_perfect and is_continuous:
            tense_features['Tense_Present_Perfect_Cont'] = 1
        elif is_perfect:
            tense_features['Tense_Present_Perfect'] = 1
        elif is_continuous:
            tense_features['Tense_Present_Continuous'] = 1
        else:
            tense_features['Tense_Simple_Present'] = 1
    
    elif is_past:
        if is_perfect and is_continuous:
            tense_features['Tense_Past_Perfect_Cont'] = 1
        elif is_perfect:
            tense_features['Tense_Past_Perfect'] = 1
        elif is_continuous:
            tense_features['Tense_Past_Continuous'] = 1
        else:
            tense_features['Tense_Simple_Past'] = 1
    
    # Final safety check: if no tense was detected at all, try to infer from root verb
    if sum(tense_features.values()) == 0 and root_verb:
        if root_verb.tag_ == 'VBD':
            tense_features['Tense_Simple_Past'] = 1
        elif root_verb.tag_ in ('VBP', 'VBZ', 'VB'):
            tense_features['Tense_Simple_Present'] = 1
    
    return tense_features

def is_passive_voice(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    is_passive = any(token.dep_ == "auxpass" for token in doc)
    return {"is_passive": int(is_passive)}

# Function to count subordinate clauses
def count_subordinate_clauses(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    count = sum(1 for token in doc if token.dep_ in CLAUSE_DEPS)
    return {"subordinate_clause_count": count}

# Number of noun phrases
def count_noun_phrases(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    count = len(list(doc.noun_chunks))
    return {"noun_phrase_count": count}

# Average Noun Phrase Length
def average_noun_phrase_length(doc: spacy.tokens.doc.Doc) -> float:
    noun_phrases = list(doc.noun_chunks)
    if not noun_phrases:
        return {"average_noun_phrase_length": 0.0}
    avg_np_length = sum(len(np) for np in noun_phrases) / len(noun_phrases)
    return {"average_noun_phrase_length": avg_np_length}

# Prepositional Phrase Count (PP Count)
def count_prepositional_phrases(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    count = sum(1 for token in doc if token.dep_ == "prep")
    return {"prepositional_phrase_count": count}

# Number of verb phrases
# Verb phrases are headed by verbs.
# So count verbs that are heads of a predicate:
# Why exclude "aux" + "auxpass"?
# Because verbs like is, was used as auxiliaries are not heads of new verb phrases.
# Example:
# "She has been running." -> 1 verb phrase (head = "running")
def count_verb_phrases(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    vp_heads = [tok for tok in doc 
            if tok.pos_ in ("VERB", "AUX") 
            and tok.dep_ not in ("aux", "auxpass")]
    return {"verb_phrase_count": len(vp_heads)}

# No. of Adjectives
def count_adjectives(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
    return {"adjective_count": adjective_count}

# No. of Adverbs
def count_adverbs(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    adverb_count = sum(1 for token in doc if token.pos_ == "ADV")
    return {"adverb_count": adverb_count}

# No. of Proper Nouns
def count_proper_nouns(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    proper_noun_count = sum(1 for token in doc if token.pos_ == "PROPN")
    return {"proper_noun_count": proper_noun_count}

def get_sentence_length(doc: spacy.tokens.doc.Doc) -> Dict[str, int]:
    length = len(doc)
    return {"sentence_length": length}

def get_features(data: pd.DataFrame, concat: bool = True) -> pd.DataFrame:
    # Process all sentences in batch
    docs = list(nlp.pipe(data['translation'], batch_size=50))

    tense_features_list = []
    passive_feature_list = []
    subordinate_clause_feature_list = []
    noun_phrase_feature_list = []
    average_np_length_feature_list = []
    prepositional_phrase_feature_list = []
    verb_phrase_feature_list = []
    adjective_count_feature_list = []
    adverb_count_feature_list = []
    proper_noun_count_feature_list = []
    sentence_length_feature_list = []


    for doc in docs:
        tense_features_list.append(extract_tense_features_from_doc(doc))
        passive_feature_list.append(is_passive_voice(doc))
        subordinate_clause_feature_list.append(count_subordinate_clauses(doc))
        noun_phrase_feature_list.append(count_noun_phrases(doc))
        average_np_length_feature_list.append(average_noun_phrase_length(doc))
        prepositional_phrase_feature_list.append(count_prepositional_phrases(doc))
        verb_phrase_feature_list.append(count_verb_phrases(doc))
        adjective_count_feature_list.append(count_adjectives(doc))
        adverb_count_feature_list.append(count_adverbs(doc))
        proper_noun_count_feature_list.append(count_proper_nouns(doc))
        sentence_length_feature_list.append(get_sentence_length(doc))

    # Convert to DataFrame and join

    tense_df = pd.DataFrame(tense_features_list)
    passive_df = pd.DataFrame(passive_feature_list)
    subordinate_clause_df = pd.DataFrame(subordinate_clause_feature_list)
    noun_phrase_df = pd.DataFrame(noun_phrase_feature_list)
    average_np_length_df = pd.DataFrame(average_np_length_feature_list)
    prepositional_phrase_df = pd.DataFrame(prepositional_phrase_feature_list)
    verb_phrase_df = pd.DataFrame(verb_phrase_feature_list)
    adjective_count_df = pd.DataFrame(adjective_count_feature_list)
    adverb_count_df = pd.DataFrame(adverb_count_feature_list)
    proper_noun_count_df = pd.DataFrame(proper_noun_count_feature_list)
    sentence_length_df = pd.DataFrame(sentence_length_feature_list)

    detailed_data = pd.concat([tense_df, passive_df, subordinate_clause_df, noun_phrase_df, average_np_length_df, prepositional_phrase_df, verb_phrase_df, adjective_count_df, adverb_count_df, proper_noun_count_df, sentence_length_df], axis=1)

    if concat:
        detailed_data = pd.concat([data, detailed_data], axis=1)
    
    return detailed_data