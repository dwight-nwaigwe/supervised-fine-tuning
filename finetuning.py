# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

# 2. Modern replacement for setup_chat_format
# Attach ChatML format template directly to the tokenizer
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)

tokenizer.chat_template = CHATML_TEMPLATE
tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})

# Ensure padding token and direction are defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Resize model vocabulary size to fit the newly added special tokens
model.resize_token_embeddings(len(tokenizer))


from transformers import pipeline
# Let's test the base model before training
prompt = "What is the primary function of mitochondria within a cell?"

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
pipe(prompt, max_new_tokens=100)
# [{'generated_text': 'What is the primary function of mitochondria within a cell?\n\The function of the mitochondria is to produce energy for the cell through a process called cellular respiration.'}]


from datasets import load_dataset

ds = load_dataset("argilla/synthetic-concise-reasoning-sft-filtered")
def tokenize_function(examples):
    examples["text"] = tokenizer.apply_chat_template([{"role": "user", "content": examples["prompt"].strip()}, {"role": "assistant", "content": examples["completion"].strip()}], tokenize=False)
    return examples
ds = ds.map(tokenize_function)


os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure the SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=100,  # Frequency of logging training metrics
    hub_model_id="argilla/SmolLM2-360M-synthetic-concise-reasoning",  # Set a unique name for your model
    push_to_hub=True,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    processing_class=tokenizer,
)
trainer.train()
# {'loss': 1.4498, 'grad_norm': 2.3919131755828857, 'learning_rate': 4e-05, 'epoch': 0.1}
# {'loss': 1.362, 'grad_norm': 1.6650595664978027, 'learning_rate': 3e-05, 'epoch': 0.19}
# {'loss': 1.3778, 'grad_norm': 1.4778285026550293, 'learning_rate': 2e-05, 'epoch': 0.29}
# {'loss': 1.3735, 'grad_norm': 2.1424977779388428, 'learning_rate': 1e-05, 'epoch': 0.39}
# {'loss': 1.3512, 'grad_norm': 2.3498542308807373, 'learning_rate': 0.0, 'epoch': 0.48}
# {'train_runtime': 1911.514, 'train_samples_per_second': 1.046, 'train_steps_per_second': 0.262, 'train_loss': 1.3828572998046875, 'epoch': 0.48}


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe(prompt, max_new_tokens=100)
# [{'generated_text': 'The primary function of mitochondria is to generate energy for the cell. They are organelles found in eukaryotic cells that convert nutrients into ATP (adenosine triphosphate), which is the primary source of energy for cellular processes.'}]

