import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from prompts import get_prompt


def eval_model(adapters_name: str = "tmp/llama-output/model") -> None:
    # Load base model
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    model = model.merge_and_unload()

    # Load dataset
    df = pd.read_excel("valid_data.xlsx")
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

    df.to_excel("valid_data.xlsx", index=False)


if __name__ == "__main__":
    from huggingface_hub import login

    login()
    eval_model()
