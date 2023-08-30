from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from custom.prompts import get_prompt
from custom.train_model import MODEL_DIR

BASE_DIR = Path(__file__).parent


def eval_model() -> None:
    # Load base model
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, str(MODEL_DIR.resolve()))
    model = model.merge_and_unload()

    # Load dataset
    fp = str((BASE_DIR / "valid_data.xlsx").resolve())
    df = pd.read_excel(fp)
    with torch.no_grad():
        model.eval()
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt = get_prompt(row["model_input"], row["statement"])
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = tokenizer.decode(
                model.generate(**input_ids, max_new_tokens=10)[0],
                skip_special_tokens=True,
            )
            df.at[index, "llama-prediction"] = output

    df.to_excel(fp, index=False)


if __name__ == "__main__":
    from huggingface_hub import login

    login()
    eval_model()
