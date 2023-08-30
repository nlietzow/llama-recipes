from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import LlamaTokenizer

from custom.prompts import get_prompt
from ft_datasets.utils import Concatenator


def get_train_dataset(tokenizer: LlamaTokenizer) -> Dataset:
    df = pd.read_excel(str((Path(__file__).parent / "train_data.xlsx").resolve()))

    prompts = []
    for index, row in df.iterrows():
        prompts.append(
            get_prompt(
                row["model_input"],
                row["statement"],
                row["prediction"],
                tokenizer.eos_token,
            )
        )

    dataset = Dataset.from_dict({"prompt": prompts})
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(
        Concatenator(),
        batched=True,
    )

    return dataset
