FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 as BUILD
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install gunicorn
RUN venv/bin/pip install -U pip wheel cython
RUN venv/bin/pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN venv/bin/pip install -r requirements.txt

COPY src src
EXPOSE 5000
ENTRYPOINT ["venv/bin/python", "app.py"]
