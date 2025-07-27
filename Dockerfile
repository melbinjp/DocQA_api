FROM python:3.10-slim
WORKDIR /app

# install torch first so that subsequent deps see it already resolved
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache

COPY . ./
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]