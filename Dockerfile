FROM python:3.10-slim

ENV TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/hf_cache

COPY . ./

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]