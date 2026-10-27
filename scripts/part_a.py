from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; DB=ROOT/'data'/'instamart.db'

def run():
    CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
    customers=pd.read_csv(RAW/'customers.csv', keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv', keep_default_na=False)
    tickets=pd.read_csv(RAW/'tickets.csv', keep_default_na=False)
    # Audit snapshots
    audit=[]
    for name,df in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        for col in df.columns:
            vals=df[col]
            missing=(vals.astype(str).str.strip()=='').sum() if name in ('customers','orders','tickets') else vals.isna().sum()
            audit.append({'table':name,'rows':len(df),'column':col,'dtype':str(df[col].dtype),'missing_count':int(missing),'missing_pct':round(100*missing/len(df),2)})
    pd.DataFrame(audit).to_csv(OUT/'raw_audit.csv',index=False)
    # Customer cleaning
    customers['city']=customers['city'].astype(str).str.strip().str.title()
    g=customers['gender'].astype(str).str.strip().str.lower()
    customers['gender']=g.map(lambda x: 'male' if x in {'m','male','man','men','boy'} else ('female' if x in {'f','female','woman','women','girl'} else 'unknown'))
    customers['age']=pd.to_numeric(customers['age'].replace('',np.nan),errors='coerce')
    customers.loc[~customers['age'].between(10,100),'age']=np.nan
    customers['age']=customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['age']=customers['age'].round().astype(int)
    customers['signup_date']=pd.to_datetime(customers['signup_date'],errors='coerce')
    # Orders clean in required order
    orders['order_ts']=pd.to_datetime(orders['order_ts'],errors='coerce')
    exact_before=len(orders); orders=orders.drop_duplicates().copy(); exact_removed=exact_before-len(orders)
    orders=orders.sort_values(['order_id','order_ts']).drop_duplicates('order_id',keep='last').copy()
    orders['quantity']=pd.to_numeric(orders['quantity'],errors='coerce')
    orders['discount_pct']=pd.to_numeric(orders['discount_pct'],errors='coerce').fillna(0)
    orders['order_amount']=pd.to_numeric(orders['order_amount'],errors='coerce')
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    miss=orders['order_amount'].isna()
    orders.loc[miss,'order_amount']=(orders.loc[miss,'price_inr']*orders.loc[miss,'quantity']*(1-orders.loc[miss,'discount_pct']/100)).round(2)
    orders['delivery_minutes']=pd.to_numeric(orders['delivery_minutes'].replace('',np.nan),errors='coerce')
    orders.loc[orders['delivery_minutes']<0,'delivery_minutes']=np.nan
    cat_median=orders.loc[orders.status.eq('completed')].groupby('category')['delivery_minutes'].median()
    cm=orders.status.eq('completed') & orders.delivery_minutes.isna()
    orders.loc[cm,'delivery_minutes']=orders.loc[cm,'category'].map(cat_median)
    orders['delivery_minutes']=orders['delivery_minutes'].round().astype('Int64')
    orders['quantity']=orders['quantity'].astype(int)
    orders['order_amount']=orders['order_amount'].round(2)
    orders['order_ts']=orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    customers['signup_date']=customers['signup_date'].dt.strftime('%Y-%m-%d')
    orders=orders.drop(columns=['price_inr','category'])
    # Stable output order
    customers.to_csv(CLEAN/'customers.csv',index=False); products.to_csv(CLEAN/'products.csv',index=False); orders.to_csv(CLEAN/'orders.csv',index=False)
    DB.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(DB) as con:
        customers.to_sql('customers',con,if_exists='replace',index=False)
        products.to_sql('products',con,if_exists='replace',index=False)
        orders.to_sql('orders',con,if_exists='replace',index=False)
    # Audit reference values
    raw_orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    raw_customers=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    ref={'raw_orders_rows':len(raw_orders),'exact_duplicates':int(raw_orders.duplicated().sum()),'pending_rows':int(raw_orders.status.eq('pending').sum()),'distinct_order_ids':int(raw_orders.order_id.nunique()),'nonnumeric_amounts':int(pd.to_numeric(raw_orders.order_amount,errors='coerce').isna().sum()),'negative_delivery':int((pd.to_numeric(raw_orders.delivery_minutes,errors='coerce')<0).sum()),'blank_delivery':int(raw_orders.delivery_minutes.astype(str).str.strip().eq('').sum()),'blank_age':int(raw_customers.age.astype(str).str.strip().eq('').sum()),'age_outside_10_100':int((pd.to_numeric(raw_customers.age,errors='coerce').notna() & ~pd.to_numeric(raw_customers.age,errors='coerce').between(10,100)).sum()),'raw_city_spellings':int(raw_customers.city.nunique()),'raw_gender_spellings_nonblank':int(raw_customers.loc[raw_customers.gender.astype(str).str.strip().ne(''),'gender'].nunique()),'blank_gender':int(raw_customers.gender.astype(str).str.strip().eq('').sum())}
    pd.DataFrame([{'metric':k,'value':v} for k,v in ref.items()]).to_csv(OUT/'raw_audit_metrics.csv',index=False)
    # SQL outputs and pandas exploration
    sql_text=(ROOT/'sql'/'queries.sql').read_text()
    statements=[s.strip() for s in sql_text.split(';') if s.strip() and not s.strip().startswith('--')]
    query_results=[]
    with sqlite3.connect(DB) as con:
        # Split by semicolon, comments are retained but valid SQLite comments
        for i,stmt in enumerate([s for s in sql_text.split(';') if s.strip()],1):
            try:
                res=pd.read_sql_query(stmt,con); res.to_csv(OUT/f'query_q{i}.csv',index=False); query_results.append({'query':f'Q{i}','rows':len(res)})
            except Exception as e: query_results.append({'query':f'Q{i}','error':str(e)})
    completed=orders[orders.status.eq('completed')].copy(); completed['order_amount']=pd.to_numeric(completed.order_amount)
    # pivot cross-check and descriptive statistics
    joined=completed.merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id')
    pivot=pd.pivot_table(joined,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True,margins_name='All')
    pivot.to_csv(OUT/'city_category_pivot.csv')
    city_stats=completed.merge(customers[['customer_id','city']],on='customer_id').groupby('city').agg(sum=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','count')).reset_index()
    city_stats.to_csv(OUT/'city_spend_stats.csv',index=False)
    x=completed[['order_amount','discount_pct','quantity','delivery_minutes']].apply(pd.to_numeric,errors='coerce')
    q1=x.order_amount.quantile(.25); q3=x.order_amount.quantile(.75); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
    desc={'mean':x.order_amount.mean(),'median':x.order_amount.median(),'skew':x.order_amount.skew(),'q1':q1,'q3':q3,'lower_fence':lo,'upper_fence':hi,'outliers_below':int((x.order_amount<lo).sum()),'outliers_above':int((x.order_amount>hi).sum())}
    pd.DataFrame([desc]).to_csv(OUT/'order_amount_descriptive.csv',index=False)
    x.corr().to_csv(OUT/'completed_order_correlations.csv')
    print(f"Part A complete: removed {exact_removed} exact duplicate rows; clean orders={len(orders)}; completed revenue INR {completed.order_amount.sum():.2f}")

if __name__=='__main__': run()
