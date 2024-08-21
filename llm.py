import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Undi95/Llama-3-LewdPlay-8B-evo"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
import ipdb;ipdb.set_trace()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
#    device_map="cpu",
    torch_dtype=dtype,
)

prompt = input('input prompt:')

chat = [
    {"role": "user", "content": prompt},
]

input_ids = tokenizer.apply_chat_template(
    chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=81920,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))

