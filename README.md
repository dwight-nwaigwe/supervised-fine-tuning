# Supervised finetuning of a large language model
This is my attempt to reproduce results from https://huggingface.co/blog/davidberenstein1957/fine-tune-a-smollm-on-synthetic-data-of-llm. I could not get the original code to work. The code here is updated to reflect some things which no longer work in the original tutorial. I also use LoRA whereas the original tutorial updates all of the model's weights. I rented GPUs from RunPod to run the code. Depending on what GPU you have, you may get a run-time error training. I had this problem and it went away after trying a different GPU. 

Note that in the original tutorial, there are the following lines:

```python
from transformers import pipeline
# Let's test the base model before training
prompt = "What is the primary function of mitochondria within a cell?"

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
pipe(prompt, max_new_tokens=100)
# [{'generated_text': 'What is the primary function of mitochondria within a cell?\n\The function of the mitochondria is to produce energy for the cell through a process called cellular respiration.'}]
```

The last line (with the comment) is misleading. It suggests that the output of the command above it will give you the commented line but this is not the case. If it did, it means that the pre-finetuned model gives you output which is the same as the data on which you want to make it finetuned! This caused me confusion until I executed this block and I saw that the output is different from the commented line, as it should be, otherwise there is no point in finetuning.

Below we paste the inference code block and its output:
```python
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

```

Q: What is the primary function of mitochondria within a cell?
A: The primary function of mitochondria in a cell is to generate energy for its operation through cellular respiration. They produce ATP (adenosine triphosphate) which is used by the cell's main energy source called glycolysis or oxidative phosphorylation. This process involves converting glucose into adenosine diphosphate (ADP), which then becomes usable energy for various cellular functions such as muscle contraction, nerve impulses, and protein synthesis. 

Q: What causes acid rain?
A: Acid rain occurs when pollutants from human activities such as industrial processes, vehicle emissions, or natural sources like lightning strike water droplets in clouds. These pollutants react with oxygen in the atmosphere to form acidic compounds.
The main cause of acid rain is:


Volatile organic compounds (VOCs) from vehicles: VOCs evaporate quickly into the air and mix with water vapor to create acids.


Sulfur dioxide (SO2) emissions from power plants: SO2 reacts with water vapor in the atmosphere to produce sulfuric acid.


Nitrogen oxides (NOx) emissions from fossil fuel combustion: NOx reacts with water vapor to form nitric acid.


Natural factors: Volcanic eruptions release large amounts of sulfur dioxide into the atmosphere, which can also contribute to acid rain.


These pollutants combine with atmospheric moisture to form acidic particles, which fall back to Earth as acid rain. 

Q: What are the main reasons for declining or revoking diplomatic recognition of a country?
A: Decline in Diplomatic Recognition:


Economic Issues: A decline in economic growth can lead to reduced government revenue, making it difficult for countries to maintain their diplomatic relationships.


Political Instability: Political instability can result from internal conflicts, corruption, or other factors that undermine trust between nations.


Humanitarian crises: Conflicts such as wars, natural disasters, or pandemics can cause significant harm to civilians and damage international relations.


Environmental degradation: Climate change, pollution, and other environmental issues can negatively impact global cooperation and diplomacy.


Geopolitical tensions: Changes in leadership or foreign policy decisions by one nation can create uncertainty about its intentions towards another country's interests.


Global events: Major world events like terrorism, pandemics, or major military conflicts can disrupt diplomatic relationships and make it challenging to maintain communication channels.


Lack of engagement: Insufficient efforts from governments or diplomats to engage with other nations may lead to a decrease in diplomatic recognition over time.


These factors contribute to why some countries might choose not to renew their diplomatic ties or revoke them altogether. 


Click to add a cell.
Simple
1
1
Python 3 (ipykernel) | Idle
Mode: Command
Ln 1, Col 1
Untitled.ipynb
1
