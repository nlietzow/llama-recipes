instructions = """
You receive a customer review and a statement. 

If the statement can be clearly inferred from the review, 
classify it as "True". 

The statement does not have to contain the same wording as the review, 
but the statement must be evident from the review. 

If this is not the case, classify the statement as "False".
"""

prompt = f"""
{" ".join(instructions.split())}

Review: {{review}}
Statement: {{statement}}
---
Label:
""".strip()


def get_prompt(
        review: str,
        statement: str,
        label: str = "",
        eos_token: str = "",
) -> str:
    prompt_filled = prompt.format(
        review=" ".join(review.split()),
        statement=" ".join(statement.split()),
    ).strip()

    if label:
        assert label in ("True", "False"), f"Invalid label: {label}"
        assert eos_token, "eos_token must be provided if label is provided"

        prompt_filled += f" {label}{eos_token}"

    return prompt_filled
