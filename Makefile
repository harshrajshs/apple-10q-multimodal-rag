install:
\tpython -m pip install -r requirements.txt

test:
\tpytest -q

ingest:
\tpython scripts/ingest.py --pdf data/2022_Q3_AAPL.pdf --out artifacts/index

run:
\tuvicorn app.api.main:app --host 0.0.0.0 --port 8000

docker-up:
\tdocker compose up --build
