from fastapi import FastAPI
from typing import Optional
from ray import serve
from ray.serve.handle import DeploymentHandle

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIIngress:
    def __init__(self, llm: DeploymentHandle):
        self.llm = llm

    @app.post("/")
    async def handle_request(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = 8192, 
        do_sample: Optional[bool] = True, 
        temperature: Optional[float] = 0.6, 
        top_p: Optional[float] = 0.9
    ) -> str:
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p
        }
        return await self.llm.translate.remote(prompt, generation_config)
