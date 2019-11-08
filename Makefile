install:
\tpip install -r requirements.txt

pipeline:
\tpython scripts/run_pipeline.py

dashboard:
\tstreamlit run dashboard/app.py

api:
\tuvicorn api.main:app --reload

test:
\tpytest -q
