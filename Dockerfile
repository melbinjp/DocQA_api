FROM python:3.10-slim
WORKDIR /app

# install torch first so that subsequent deps see it already resolved
COPY requirements.txt ./
# The CPU index no longer hosts every torch dependency as a wheel, so deps fall
# back to PyPI; torch itself still resolves to the +cpu wheel because a local
# version sorts above the plain release. Pinned at 2.9.1 rather than 2.4.0:
# current transformers only imports torch behind a >=2.5 version guard, so 2.4
# dies at import time with "NameError: name 'torch' is not defined".
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_DOWNLOAD_TIMEOUT=120

COPY . ./
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]