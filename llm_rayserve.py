import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import ray
from ray import serve
from fastapi import FastAPI

app = FastAPI()

model_id = "Undi95/Llama-3-LewdPlay-8B-evo"
dtype = torch.bfloat16

from huggingface_hub import snapshot_download
snapshot_download(repo_id=model_id, ignore_patterns=["*.gguf"])  # Download our BF16 model without downloading GGUF models.

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
@serve.ingress(app)
class Llm:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=dtype,
        )

    @app.post("/")
    def translate(self, prompt: str) -> str:
        chat = [
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1] :]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)
        return answer


llm_app = Llm.bind()

