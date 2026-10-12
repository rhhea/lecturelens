from functools import lru_cache
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"


# Load the model only once and reuse it
@lru_cache(maxsize=1)
def _get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


def _generate(
    prompt: str,
    max_new_tokens: int = 300,
    min_new_tokens: int = 80,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.2,
) -> str:

    tokenizer, model = _get_model()

    # Convert prompt into tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample and temperature is not None:
        generation_args["temperature"] = temperature

    # Generate summary
    output_ids = model.generate(
        **inputs,
        **generation_args,
    )

    # Convert tokens back into text
    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result.strip()


def summarize_passage(
    passage: str,
    max_new_tokens: int = 300,
) -> str:

    if not passage or not passage.strip():
        return ""

    prompt = f"""
Summarize the following passage in 2 to 3 clear paragraphs.

Requirements:
- Write 2 to 3 paragraphs.
- Each paragraph should contain 2 to 4 complete sentences.
- Include the important concepts, facts, explanations, and conclusions.
- Preserve the meaning of the original passage.
- Use simple and clear language.
- Do not use bullet points.
- Do not use headings.
- Do not repeat the same information.

Passage:
{passage}
"""

    return _generate(
        prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=80,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )


def summarize_document(
    passages: List[str],
    max_new_tokens: int = 300,
) -> str:

    if not passages:
        return ""

    # Remove empty passages
    passages = [
        passage.strip()
        for passage in passages
        if passage and passage.strip()
    ]

    if not passages:
        return ""

    # Summarize each passage first
    passage_summaries = [
        summarize_passage(
            passage,
            max_new_tokens=max_new_tokens,
        )
        for passage in passages
    ]

    # If there is only one passage, return its summary
    if len(passage_summaries) == 1:
        return passage_summaries[0]

    # Combine the individual summaries
    combined = "\n\n".join(passage_summaries)

    # Generate the final summary
    prompt = f"""
Create a final summary of the following lecture/document content.

Requirements:
- Write exactly 2 to 3 paragraphs.
- Each paragraph should contain 2 to 4 complete sentences.
- Cover the main concepts and important details.
- Explain the key ideas clearly.
- Remove unnecessary repetition.
- Preserve important technical information.
- Use simple, readable language.
- Do not use bullet points.
- Do not use headings.

Lecture/document content:
{combined}
"""

    return _generate(
        prompt,
        max_new_tokens=max_new_tokens,
        min_new_tokens=100,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )