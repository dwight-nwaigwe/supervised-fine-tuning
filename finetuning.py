import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
output_dir = "./smollm2-concise-reasoning-lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

dataset = load_dataset("argilla/synthetic-concise-reasoning-sft-filtered", split="train")

def to_prompt_completion(ex):
    return {
        "prompt": [
            {"role": "system", "content": ex["system_prompt"]},
            {"role": "user", "content": ex["prompt"]},
        ],
        "completion": [{"role": "assistant", "content": ex["completion"]}],
    }

train_ds = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)

sft_config = SFTConfig(
    output_dir=output_dir,
    max_length=512,
    packing=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,             
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    bf16=use_bf16,
    fp16=not use_bf16,
    # completion_only_loss defaults to True for prompt-completion data
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(output_dir)

system_prompt = (
    "You are an AI assistant who provides brief and to-the-point responses "
    "with logical step-by-step reasoning. Your purpose is to offer straightforward "
    "explanations and answers so that you can get to the heart of the issue. "
    "Respond with extremely concise, direct justifications and evidence-based conclusions."
)
user_query = "What is the primary function of mitochondria within a cell?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
]

trainer.model.eval()
trainer.model.config.use_cache = True

questions = [
    "What is the primary function of mitochondria within a cell?",
    "What causes acid rain?",
    "What are the main reasons for declining or revoking diplomatic recognition of a country?",
]

batch = [
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    for q in questions
]

tokenizer.padding_side = "left"   # needed for batched generation; training used "right"

outs = generator(
    batch,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    batch_size=3,
)

for q, o in zip(questions, outs):
    print("Q:", q)
    print("A:", o[0]["generated_text"].strip(), "\n")
