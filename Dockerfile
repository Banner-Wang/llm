# File name: Dockerfile
FROM rayproject/ray:2.31.0

# Set the working dir for the container to /llm_serve_app
WORKDIR /llm_serve_app

# Copies the local `fake.py` file into the WORKDIR
COPY llm_rayserve.py /llm_serve_app/llm_rayserve.py

COPY download_model.py /llm_serve_app/download_model.py

RUN pip install transformers==4.44.0 torch==2.3.1 fastapi==0.111.0 accelerate==0.33.0

# download model
RUN python /llm_serve_app/download_model.py

