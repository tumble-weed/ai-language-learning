from rapidfuzz import fuzz
from pydub import AudioSegment
from pathlib import Path
import os

# This code is able to generate audio for a sentence which is divided 
# into multiple chunks in ASR

# But failing to generate audio for the sentence which stops in between a chunk

# For Example, there is a chunk which combines two sentences,
# The transcribed text we got is one of those sentences.
# In that case, the code is not able to generate audio for that sentence.

def align_sentences_to_timestamps(chunks, final_sentences, original_audio_file, threshold=80):
    """
    Aligns final cleaned sentences to original transcription chunks 
    and estimates start/end timestamps using a robust chunk-mapping method.
    
    This version interpolates timestamps *within* chunks for greater accuracy.
    
    Args:
        chunks (list): ASR output, e.g., 
            [{'text': '...', 'start': 0.5, 'end': 1.0}, ...]
        final_sentences (list): List of clean sentences, e.g.,
            ["Sentence one.", "Sentence two."]
        original_audio_file (str or Path): Path to the source audio file.
        threshold (int): Fuzzy matching score cutoff (0-100).
            
    Returns:
        list: A list of dictionaries with aligned sentences and timestamps.
    """
    
    # --- 1. Build the Combined Text and Character-to-Chunk Map ---
    
    if not chunks:
        print("⚠️ No chunks provided. Returning empty list.")
        return []

    combined_text_list = []
    char_to_chunk_map = []      # Stores the *index* of the chunk for each char
    char_start_in_chunk_map = [] # NEW: Stores global start index of each chunk's text
    current_global_index = 0    # NEW: Tracks position in combined_text
    
    for chunk_index, c in enumerate(chunks):
        text = c.get("text", "")
        
        # Add a space between chunks (except for the first one)
        if chunk_index > 0:
            combined_text_list.append(" ")
            # Map the space to the chunk it *precedes*
            char_to_chunk_map.append(chunk_index) 
            current_global_index += 1 # NEW: Account for the space
        
        # NEW: Store the global start index for this chunk's text
        char_start_in_chunk_map.append(current_global_index) 
        
        combined_text_list.append(text)
        # Map each character in the text to its chunk index
        char_to_chunk_map.extend([chunk_index] * len(text))
        current_global_index += len(text) # NEW: Add text length
        
    combined_text = "".join(combined_text_list)
    
    if not combined_text:
        print("⚠️ Chunks resulted in empty text. Returning empty list.")
        return []

    # --- 2. Align Sentences using Fuzzy Matching ---
    
    results = []
    print(f"Aligning {len(final_sentences)} sentences...")

    for sent in final_sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # Find the best partial match for the sentence in the *entire* combined text
        # This returns an object with start/end indices of the match
        match = fuzz.partial_ratio_alignment(sent, combined_text, score_cutoff=threshold)
        
        if not match:
            print(f"⚠️ Could not match: {sent[:40]}... (score < {threshold})")
            continue
            
        # Get the character indices from the match
        # match.dest_start is the start index in combined_text
        # match.dest_end is the *exclusive* end index
        
        start_char_idx = match.dest_start
        end_char_idx = match.dest_end - 1 # Make it an *inclusive* index
        
        # Ensure indices are valid
        if start_char_idx < 0 or end_char_idx >= len(char_to_chunk_map):
            print(f"⚠️ Match indices out of bounds: {sent[:40]}...")
            continue
            
        # --- 3. Get Timestamps from the Chunk Map (with Interpolation) ---
        
        # Find which chunk the *first* character belongs to
        start_chunk_idx = char_to_chunk_map[start_char_idx]
        
        # Find which chunk the *last* character belongs to
        end_chunk_idx = char_to_chunk_map[end_char_idx]
        
        
        # --- NEW LOGIC: Calculate Start Time by Interpolation ---
        start_chunk = chunks[start_chunk_idx]
        start_chunk_text = start_chunk.get("text", "")
        start_chunk_text_len = len(start_chunk_text)
        chunk_global_start_idx = char_start_in_chunk_map[start_chunk_idx]
        
        # Find the character's *local* index (relative to its chunk's text)
        # Use max(0, ...) to handle cases where the match starts on the preceding space
        char_local_index = max(0, start_char_idx - chunk_global_start_idx)

        if start_chunk_text_len == 0:
            start_time = start_chunk["start"] # Fallback for empty chunks
        else:
            chunk_duration = start_chunk["end"] - start_chunk["start"]
            time_per_char = chunk_duration / start_chunk_text_len
            start_time = start_chunk["start"] + (char_local_index * time_per_char)

            
        # --- NEW LOGIC: Calculate End Time by Interpolation ---
        end_chunk = chunks[end_chunk_idx]
        end_chunk_text = end_chunk.get("text", "")
        end_chunk_text_len = len(end_chunk_text)
        chunk_global_start_idx = char_start_in_chunk_map[end_chunk_idx]

        # Find the character's *local* index
        char_local_index = max(0, end_char_idx - chunk_global_start_idx)
        
        if end_chunk_text_len == 0:
            end_time = end_chunk["end"] # Fallback for empty chunks
        else:
            # For the *end* time, use (local_index + 1) to get the time at the *end* of that character
            char_local_index_end_point = char_local_index + 1
            
            chunk_duration = end_chunk["end"] - end_chunk["start"]
            time_per_char = chunk_duration / end_chunk_text_len
            end_time = end_chunk["start"] + (char_local_index_end_point * time_per_char)
            
            # Clamp the time to the chunk's actual end time, just in case
            end_time = min(end_time, end_chunk["end"])

        
        results.append({
            "sentence": sent,
            "start": round(start_time, 2),
            "end": round(end_time, 2)
        })
        
    print(f"✅ Successfully aligned {len(results)} sentences.")

    # --- 4. Export Audio Segments ---
    
    if not results:
        print("No aligned sentences to export.")
        return results

    print(f"\n🎧 Exporting {len(results)} audio segments...")
    
    try:
        audio = AudioSegment.from_file(original_audio_file)
    except FileNotFoundError:
        print(f"❌ ERROR: Audio file not found at {original_audio_file}")
        return results
    except Exception as e:
        print(f"❌ ERROR: Could not load audio file. Is ffmpeg installed? {e}")
        return results
        
    # Create a dedicated output directory
    audio_path = Path(original_audio_file)
    output_dir = audio_path.parent / f"{audio_path.stem}_segments"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving segments to: {output_dir}")

    for i, r in enumerate(results):
        start_ms = int(r["start"])
        end_ms = int(r["end"])
        
        if start_ms >= end_ms:
             print(f"⚠️ Skipping segment with invalid times: start={start_ms}ms, end={end_ms}ms for '{r['sentence'][:20]}...'")
             continue

        segment = audio[start_ms:end_ms]
        
        # Using the filename format from your code
        output_filename = output_dir / f"segment_{i+1}.wav"
        
        try:
            segment.export(output_filename, format="wav")
        except Exception as e:
            print(f"❌ ERROR exporting {output_filename}: {e}")
        
        # Storing the path in the result dictionary, as in your code
        r['audio_file'] = str(output_filename)
    
    print("✅ Export complete.")
    return results