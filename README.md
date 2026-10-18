# Supervised tuning of a large language model
This is my attempt to reproduce results from https://huggingface.co/blog/davidberenstein1957/fine-tune-a-smollm-on-synthetic-data-of-llm. The code here is updated to reflect some things which no longer work in the original tutorial. I rented GPUs from RunPod to do this tutorial. Depending on what GPU you have, you may get a run-time error training. I had this problem and it went away after trying a different GPU. 

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

