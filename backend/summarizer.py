from functools import lru_cache
from typing import List

MODEL_NAME = "google/flan-t5-base"


@lru_cache(maxsize=1)
def _get_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def _generate(prompt: str, max_new_tokens: int = 120, min_new_tokens: int = 20,
              do_sample: bool = False, temperature: float = None,
              num_beams: int = 1, no_repeat_ngram_size: int = 0,
              repetition_penalty: float = 1.0) -> str:
    tokenizer, model = _get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def summarize_passage(passage: str, max_new_tokens: int = 120) -> str:
    return _generate(
        f"summarize: {passage}",
        max_new_tokens=max_new_tokens,
        min_new_tokens=25,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.4,
    )

def summarize_document(passages: List[str], max_new_tokens: int = 120) -> str:
    if not passages:
        return ""
    passage_summaries = [summarize_passage(p, max_new_tokens=max_new_tokens) for p in passages]
    if len(passage_summaries) == 1:
        return passage_summaries[0]
    combined = " ".join(passage_summaries)
    return summarize_passage(combined, max_new_tokens=max_new_tokens)