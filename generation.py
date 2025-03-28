# generation.py
import logging
import json
import time
from typing import Tuple
from openai import OpenAIError

# Import shared client, tokenizer, utils and constants
# Ensure ALL constants used within this module are imported
from utils import client, tokenizer, count_tokens, logger
from constants import (
    GENERATION_MODEL, MAX_MODEL_TOKENS_SUMMARY, CHUNK_SIZE, CHUNK_OVERLAP,
    SYSTEM_PROMPT_GENERATION, USER_PROMPT_GENERATION,
    SYSTEM_PROMPT_COMBINE, USER_PROMPT_COMBINE,
    # Status constants used in this module's functions:
    STATUS_STEP_2_SINGLE, STATUS_STEP_2_CHUNK, STATUS_STEP_2_MAP, STATUS_STEP_2_REDUCE,
    # Error constants used in this module:
    ERROR_GENERATE_CONTENT_CHUNK, ERROR_GENERATE_CONTENT_COMBINE,
    ERROR_GENERATE_CONTENT_GENERAL
)

# No need to get logger again if imported from utils

def _generate_content_single_call(transcript: str) -> str:
    """Helper for single API call (Synchronous)."""
    if client is None: raise RuntimeError("OpenAI client not initialized.")
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GENERATION},
            {"role": "user", "content": USER_PROMPT_GENERATION.format(transcript)}
        ],
        temperature=0.5,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    if not content:
        logger.error("OpenAI API returned empty content for single call.")
        raise ValueError("LLM returned empty content.")
    return content

def _generate_content_map_reduce(transcript: str, status_update_func) -> str:
    """Helper for Map-Reduce (Synchronous)."""
    if tokenizer is None: raise RuntimeError("Tokenizer not initialized.")
    if client is None: raise RuntimeError("OpenAI client not initialized.")

    # 1. Chunking
    tokens = tokenizer.encode(transcript)
    chunks: list[str] = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + CHUNK_SIZE, len(tokens))
        try:
            chunk_text = tokenizer.decode(tokens[start_index:end_index])
            chunks.append(chunk_text)
        except Exception as e:
            logger.error(f"Error decoding token chunk {start_index}:{end_index}. Skipping. Error: {e}")
        start_index += CHUNK_SIZE - CHUNK_OVERLAP
        if start_index >= len(tokens): break

    if not chunks:
        raise RuntimeError("Failed to create any valid text chunks from the transcript.")
    logger.info(f"Split transcript into {len(chunks)} chunks.")

    # 2. Map Step
    chunk_content_jsons: list[str] = []
    for i, chunk in enumerate(chunks):
        if status_update_func:
            # Use the imported constant
            status_update_func(label=STATUS_STEP_2_MAP.format(i=i+1, n=len(chunks)))
        logger.info(f"Map: Processing chunk {i+1}/{len(chunks)}")
        try:
            chunk_json_str = _generate_content_single_call(chunk) # Direct synchronous call
            try:
                json.loads(chunk_json_str) # Check if valid JSON
                chunk_content_jsons.append(chunk_json_str)
                logger.debug(f"Chunk {i+1} generated valid JSON.")
            except json.JSONDecodeError:
                logger.warning(f"Chunk {i+1} response was not valid JSON, skipping.")
                continue # Skip this chunk's output if it's not valid JSON
        except ValueError as ve: # Catch empty content error from helper
            logger.error(f"Chunk {i+1} generation returned empty content, skipping. Error: {ve}")
            continue # Skip chunk if LLM returns empty
        except Exception as e: # Catch other API or processing errors
            logger.error(ERROR_GENERATE_CONTENT_CHUNK.format(i=i+1, n=len(chunks), e=e), exc_info=True)
            logger.warning(f"Skipping chunk {i+1} due to error during generation.")
            # Continue processing other chunks for robustness

    if not chunk_content_jsons:
        raise RuntimeError("Map phase failed: No valid JSON content generated from any chunk.")

    # 3. Reduce Step
    if status_update_func:
        # Use the imported constant
        status_update_func(label=STATUS_STEP_2_REDUCE)
    logger.info("Reduce: Combining chunk content.")
    combined_input_str = "\n\n".join(chunk_content_jsons) # Join JSON strings
    combined_tokens = count_tokens(SYSTEM_PROMPT_COMBINE + USER_PROMPT_COMBINE.format(combined_input_str))
    if combined_tokens > MAX_MODEL_TOKENS_SUMMARY:
        logger.warning(f"Combined input for Reduce ({combined_tokens} tokens) may exceed model limit ({MAX_MODEL_TOKENS_SUMMARY}). Combining might fail or truncate.")

    try:
        final_content_json = client.chat.completions.create( # Direct synchronous call
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_COMBINE},
                {"role": "user", "content": USER_PROMPT_COMBINE.format(combined_input_str)}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        ).choices[0].message.content
        if not final_content_json: # Check for empty content
            logger.error("OpenAI API returned empty content during Reduce step.")
            raise ValueError("LLM returned empty content during Reduce phase.")
        return final_content_json
    except ValueError as ve: # Catch empty content error
        logger.error(f"Reduce step returned empty content. Error: {ve}")
        raise RuntimeError(f"Reduce step failed: {ve}") from ve
    except Exception as e: # Catch other API or processing errors
        logger.error(ERROR_GENERATE_CONTENT_COMBINE.format(e=e), exc_info=True)
        raise RuntimeError(ERROR_GENERATE_CONTENT_COMBINE.format(e=e))

def generate_content(transcript: str, status_update_func) -> Tuple[str, float]:
    """Generates summary and flashcards using OpenAI API (Synchronous)."""
    start_time = time.time()
    try:
        # Check dependencies
        if tokenizer is None: raise RuntimeError("Tokenizer not initialized.")
        if client is None: raise RuntimeError("OpenAI client not initialized.")

        # Calculate token counts
        prompt_tokens = count_tokens(SYSTEM_PROMPT_GENERATION + USER_PROMPT_GENERATION.format(""))
        transcript_tokens = count_tokens(transcript)
        if prompt_tokens == -1 or transcript_tokens == -1:
             raise RuntimeError("Failed to count tokens for generation estimate.")
        total_tokens_single_call = prompt_tokens + transcript_tokens
        logger.info(f"Cleaned transcript token count: {transcript_tokens}")
        logger.info(f"Estimated single call tokens: {total_tokens_single_call} vs Max: {MAX_MODEL_TOKENS_SUMMARY}")

        # Decide on generation strategy
        if total_tokens_single_call <= MAX_MODEL_TOKENS_SUMMARY:
            if status_update_func:
                # Use the imported constant
                status_update_func(label=STATUS_STEP_2_SINGLE)
            logger.info("Processing content generation in single call.")
            content_json = _generate_content_single_call(transcript)
        else:
            logger.warning("Transcript exceeds token limit, initiating Map-Reduce process.")
            # Estimate number of chunks for status update
            num_chunks_est = 1
            if (CHUNK_SIZE - CHUNK_OVERLAP) > 0 :
                 num_chunks_est= (transcript_tokens // (CHUNK_SIZE - CHUNK_OVERLAP)) + 1
            if status_update_func:
                # Use the imported constant
                status_update_func(label=STATUS_STEP_2_CHUNK.format(num_chunks=num_chunks_est))
            content_json = _generate_content_map_reduce(transcript, status_update_func)

        generation_time = round(time.time() - start_time, 2)
        logger.info(f"Content generation completed in {generation_time}s.")

        # Final check on generated content
        if not content_json or not isinstance(content_json, str):
             logger.error("Final generated content is empty or not a string after processing.")
             raise ValueError("LLM generation resulted in empty or invalid final content.")
        return content_json, generation_time

    except (OpenAIError, ValueError, RuntimeError) as e: # Catch specific known errors
         # Log the specific error type and message
         logger.error(f"{type(e).__name__} in generate_content: {e}", exc_info=True)
         # Re-raise as RuntimeError for consistent handling in pipeline
         raise RuntimeError(ERROR_GENERATE_CONTENT_GENERAL.format(e)) from e
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error during content generation: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected content gen error: {e}") from e