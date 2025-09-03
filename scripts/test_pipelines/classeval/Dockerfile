FROM python:3.10
WORKDIR /var/app

RUN mkdir -p scripts/test_pipelines/classeval/
RUN mkdir -p data/classeval/raw/
RUN mkdir -p data/classeval/post_test/

COPY ./scripts/test_pipelines/classeval/evaluation.py scripts/test_pipelines/classeval/
COPY ./scripts/test_pipelines/classeval/test_pipeline_classeval.py scripts/test_pipelines/classeval/
COPY ./data/classeval/raw/results_ClassEval*.json data/classeval/raw/
COPY ./data/classeval/sampled_tasks.csv data/classeval/
COPY ./data/classeval/ClassEval_data.json data/classeval/
COPY ./data/classeval/rq.txt scripts/test_pipelines/classeval/

WORKDIR /var/app/scripts/test_pipelines/classeval
RUN pip install --upgrade pip && pip install --no-cache-dir -r rq.txt

ENTRYPOINT ["python", "./evaluation.py"]
