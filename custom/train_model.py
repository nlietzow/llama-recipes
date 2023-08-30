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

from .make_datasets import get_train_dataset


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
    output_dir = "tmp/llama-output"
    config = {
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_checkpointing": False,
    }
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,
        logging_dir=f"{output_dir}/logs",
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
    model_dir = f"{output_dir}/model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Save merged model
    merged_model_dir = f"{output_dir}/merged_model"
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)

    return output_dir


if __name__ == '__main__':
    from huggingface_hub import login

    login()
    run_training()
