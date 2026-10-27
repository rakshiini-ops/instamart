from pathlib import Path
import sqlite3, pandas as pd, json
ROOT=Path(__file__).resolve().parent
orders=pd.read_csv(ROOT/'data'/'clean'/'orders.csv')
customers=pd.read_csv(ROOT/'data'/'clean'/'customers.csv')
products=pd.read_csv(ROOT/'data'/'clean'/'products.csv')
assert len(customers)==1200 and len(products)==40
assert len(orders)==12309 and orders.order_id.nunique()==12309
assert set(orders.status)=={'completed','cancelled'}
assert int((orders.status=='completed').sum())==11110
assert int((orders.status=='cancelled').sum())==1199
assert pd.to_numeric(orders.loc[orders.status=='completed','delivery_minutes'],errors='coerce').isna().sum()==0
assert pd.to_numeric(orders.loc[orders.status=='cancelled','delivery_minutes'],errors='coerce').isna().sum()==1199
assert pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()==0
assert round(orders.loc[orders.status=='completed','order_amount'].sum(),2)==2998668.80
assert customers.age.isna().sum()==0 and customers.age.between(10,100).all()
with sqlite3.connect(ROOT/'data'/'instamart.db') as con:
    names={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {'customers','products','orders'}<=names
for f in ['customer_segments.csv','retention_list.csv','triage_results.csv','triage_budget_demo.csv','agent_runs.jsonl','refund_ledger.jsonl']:
    p=ROOT/'outputs'/f
    assert p.exists(),f'Missing output {f}'
print('ALL CHECKPOINTS PASSED')
