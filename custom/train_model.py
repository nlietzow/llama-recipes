import pathlib
from contextlib import nullcontext

import torch
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
)

from custom.make_datasets import get_train_dataset

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "llama-output"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = OUTPUT_DIR / "model"
MERGED_MODEL_DIR = OUTPUT_DIR / "merged_model"


def run_training() -> str:
    # Load base model
    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare int-8 model for training
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model.train()
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Get dataset
    train_data = get_train_dataset(tokenizer)

    # Define train config
    config = {
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_checkpointing": False,
    }
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR.resolve()),
        overwrite_output_dir=True,
        bf16=True,
        logging_dir=str(LOG_DIR.resolve()),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=-1,
        **config,
    )

    # Run training
    profiler = nullcontext()
    with profiler:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            data_collator=default_data_collator,
            callbacks=[],
        )
        trainer.train()

    # Save model
    model.save_pretrained(str(MODEL_DIR.resolve()))
    tokenizer.save_pretrained(str(MODEL_DIR.resolve()))

    # Save merged model
    model = model.merge_and_unload()
    model.save_pretrained(str(MERGED_MODEL_DIR.resolve()))
    tokenizer.save_pretrained(str(MERGED_MODEL_DIR.resolve()))

    return str(OUTPUT_DIR.resolve())
