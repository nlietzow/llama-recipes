from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from custom.prompts import get_prompt
from custom.train_model import MODEL_DIR


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

    # Load dataset
    fp = str((Path(__file__).parent / "valid_data.xlsx").resolve())
    df = pd.read_excel(fp).head(20)
    with torch.no_grad():
        model.eval()
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt = get_prompt(row["model_input"], row["top_topic"])
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = tokenizer.decode(
                model.generate(**input_ids, max_new_tokens=10)[0],
                skip_special_tokens=True,
            )
            df.at[index, "llama-prediction"] = output

    df.to_excel(fp, index=False)
