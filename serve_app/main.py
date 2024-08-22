from api import APIIngress
from llm_model import Llm

llm_ray = APIIngress.bind(Llm.bind())
