from typing import List

from summarizer import _generate


def generate_followup_questions(passage: str, num_questions: int = 3) -> List[str]:
    prompts = [
        f"generate a follow-up question to test understanding of: {passage}",
        f"generate a question about a key detail in: {passage}",
        f"generate a question that checks a common misconception about: {passage}",
    ]

    questions = []
    for prompt in prompts[:num_questions]:
        q = _generate(
            prompt,
            max_new_tokens=64,
            min_new_tokens=8,
            do_sample=True,
            temperature=0.8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
        if q and q not in questions:
            questions.append(q)

    return questions