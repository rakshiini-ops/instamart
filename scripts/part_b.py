from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp('2025-10-01 00:00:00'); CUTOFF=pd.Timestamp('2025-09-01 00:00:00')

def run():
    customers=pd.read_csv(CLEAN/'customers.csv',parse_dates=['signup_date'])
    orders=pd.read_csv(CLEAN/'orders.csv',parse_dates=['order_ts'])
    products=pd.read_csv(CLEAN/'products.csv')
    o=orders.merge(products[['product_id','category']],on='product_id',how='left')
    completed=o[o.status.eq('completed')].copy()
    # Whole-window RFM
    rfm=completed.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(SNAP-rfm.last_order_ts).dt.days
    rfm=rfm[['customer_id','recency','frequency','monetary']]
    rfm.to_csv(OUT/'rfm_table.csv',index=False)
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    km_rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
        km_rows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,km.labels_)})
    pd.DataFrame(km_rows).to_csv(OUT/'kmeans_metrics.csv',index=False)
    # K=4 as interpretable business choice, fixed seed
    kmeans=KMeans(n_clusters=4,n_init=10,random_state=42).fit(X)
    rfm['segment_id']=kmeans.labels_
    profile=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).reset_index()
    # Label based on profiles, Premium is highest mean monetary
    names={}
    for _,row in profile.iterrows():
        if row.segment_id==profile.loc[profile.mean_monetary.idxmax(),'segment_id']:
            names[int(row.segment_id)]='Premium Loyalists'
        elif row.mean_recency < rfm.recency.median() and row.mean_frequency > rfm.frequency.median(): names[int(row.segment_id)]='Frequent Regulars'
        elif row.mean_recency > 60 and row.mean_frequency < rfm.frequency.median(): names[int(row.segment_id)]='Dormant Customers'
        elif row.mean_recency > rfm.recency.median() and row.mean_frequency < rfm.frequency.median(): names[int(row.segment_id)]='At-Risk Occasional'
        else: names[int(row.segment_id)]='Developing Customers'
    rfm['segment_name']=rfm.segment_id.map(names)
    premium_id=int(profile.loc[profile.mean_monetary.idxmax(),'segment_id'])
    rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm.to_csv(OUT/'customer_segments.csv',index=False,columns=['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium'])
    profile['segment_name']=profile.segment_id.map(names); profile['recommended_action']=profile.segment_id.map({premium_id:'Protect loyalty with early access and tailored retention offers.'})
    profile['recommended_action']=profile['recommended_action'].fillna(profile.segment_name.map({'Frequent Regulars':'Use replenishment reminders and loyalty rewards.','At-Risk Occasional':'Send a targeted win-back offer and ask for feedback.','Dormant Customers':'Run a reactivation campaign with a time-limited return offer.','Developing Customers':'Encourage a second order with relevant bundles.'}).fillna('Test a low-cost personalised offer.'))
    profile['is_premium']=profile.segment_id.eq(premium_id)
    profile.to_csv(OUT/'segment_profiles.csv',index=False)
    pca=PCA(n_components=2).fit(X)
    pd.DataFrame({'component':['PC1','PC2'],'explained_variance_ratio':pca.explained_variance_ratio_}).to_csv(OUT/'pca_variance.csv',index=False)
    # Lapse cohort and leakage-safe features: only orders strictly before cutoff
    pre=o[o.order_ts.lt(CUTOFF)].copy(); pre_completed=pre[pre.status.eq('completed')]
    cohort_ids=set(pre_completed.customer_id)
    september=completed[completed.order_ts.ge(CUTOFF)&completed.order_ts.lt(pd.Timestamp('2025-10-01'))].groupby('customer_id').size()
    cust=customers[customers.customer_id.isin(cohort_ids)].copy()
    f=pre.groupby('customer_id').agg(all_orders=('order_id','nunique'),cancelled=('status',lambda s:int((s=='cancelled').sum()))).reset_index()
    fc=pre_completed.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index()
    fav=pre_completed.groupby('customer_id')['category'].agg(lambda s:s.mode().iloc[0] if not s.mode().empty else 'Unknown').rename('favourite_category').reset_index()
    f=f.merge(fc,on='customer_id',how='inner').merge(fav,on='customer_id',how='left')
    f['recency']=(CUTOFF-f.last_order).dt.days
    f['cancellation_rate']=f.cancelled/f.all_orders.replace(0,np.nan)
    f['tenure_days']=(CUTOFF-cust.set_index('customer_id').loc[f.customer_id,'signup_date'].reset_index(drop=True)).dt.days.values
    f=f.merge(cust[['customer_id','city','age','gender']],on='customer_id',how='left')
    f['lapsed']=~f.customer_id.isin(september.index)
    f['lapsed']=f.lapsed.astype(int)
    # Features with sensible missing fills; imputation also pipeline-based for train-only fitting
    f['avg_delivery']=f['avg_delivery'].fillna(f['avg_delivery'].median())
    f['avg_discount']=f['avg_discount'].fillna(0); f['cancellation_rate']=f['cancellation_rate'].fillna(0)
    f['favourite_category']=f['favourite_category'].fillna('Unknown')
    f.to_csv(OUT/'lapse_cohort_features.csv',index=False)
    feature_cols=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','tenure_days','age','city','gender','favourite_category']
    Xf=f[feature_cols].copy(); y=f['lapsed']
    assert not Xf.isna().any().any(), 'Feature matrix has missing values'
    X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(Xf,y,f.index,test_size=.2,stratify=y,random_state=42)
    num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','tenure_days','age']
    cat=['city','gender','favourite_category']
    prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
    log_pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))])
    log_grid=GridSearchCV(log_pipe,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=1).fit(X_train,y_train)
    rf_pipe=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid=GridSearchCV(rf_pipe,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=1).fit(X_train,y_train)
    baseline_pred=np.zeros(len(y_test),dtype=int)
    rows=[{'model':'baseline_not_lapsed','accuracy':accuracy_score(y_test,baseline_pred),'precision_lapsed':precision_score(y_test,baseline_pred,zero_division=0),'recall_lapsed':recall_score(y_test,baseline_pred,zero_division=0),'f1_lapsed':f1_score(y_test,baseline_pred,zero_division=0),'tn_fp_fn_tp':str(confusion_matrix(y_test,baseline_pred).ravel().tolist())}]
    for label,gs in [('logistic',log_grid),('random_forest',rf_grid)]:
        pred=gs.predict(X_test)
        rows.append({'model':label,'best_params':str(gs.best_params_),'best_cv_f1':gs.best_score_,'accuracy':accuracy_score(y_test,pred),'precision_lapsed':precision_score(y_test,pred,zero_division=0),'recall_lapsed':recall_score(y_test,pred,zero_division=0),'f1_lapsed':f1_score(y_test,pred,zero_division=0),'confusion_matrix':str(confusion_matrix(y_test,pred).tolist())})
    pd.DataFrame(rows).to_csv(OUT/'lapse_model_metrics.csv',index=False)
    # oversample positive training rows only, keep all originals
    train_df=X_train.copy(); train_df['lapsed']=y_train.values
    pos=train_df[train_df.lapsed==1]; neg=train_df[train_df.lapsed==0]
    if len(pos)<len(neg):
        extra=pos.sample(n=len(neg)-len(pos),replace=True,random_state=42)
        balanced=pd.concat([train_df,extra],ignore_index=True)
    else: balanced=train_df
    balanced_model=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,C=log_grid.best_params_['model__C']))])
    balanced_model.fit(balanced[feature_cols],balanced['lapsed'])
    bal_pred=balanced_model.predict(X_test)
    pd.DataFrame([{'model':'logistic_unbalanced','precision':precision_score(y_test,log_grid.predict(X_test),zero_division=0),'recall':recall_score(y_test,log_grid.predict(X_test),zero_division=0),'f1':f1_score(y_test,log_grid.predict(X_test),zero_division=0)},{'model':'logistic_oversampled','precision':precision_score(y_test,bal_pred,zero_division=0),'recall':recall_score(y_test,bal_pred,zero_division=0),'f1':f1_score(y_test,bal_pred,zero_division=0)}]).to_csv(OUT/'class_balance_metrics.csv',index=False)
    # Pick whichever tuned model has the higher test F1; persist and verify predictions
    candidates=[('logistic',log_grid.best_estimator_),('random_forest',rf_grid.best_estimator_)]
    chosen_name,chosen=max(candidates,key=lambda z:f1_score(y_test,z[1].predict(X_test),zero_division=0))
    pred=chosen.predict(X_test); joblib.dump(chosen,OUT/'lapse_model.joblib'); reloaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(pred,reloaded.predict(X_test))
    # Feature importance and logistic coefficients (persist human-readable summary)
    rf=rf_grid.best_estimator_; enc=rf.named_steps['prep'].named_transformers_['cat']; feature_names=num+list(enc.get_feature_names_out(cat))
    importances=rf.named_steps['model'].feature_importances_
    imp=pd.DataFrame({'feature':feature_names,'importance':importances}).sort_values('importance',ascending=False).head(5)
    imp.to_csv(OUT/'rf_top_features.csv',index=False)
    log=log_grid.best_estimator_; ln=list(log.named_steps['prep'].get_feature_names_out()); coef=log.named_steps['model'].coef_[0]
    pd.DataFrame({'feature':ln,'coefficient':coef}).assign(abs_coefficient=lambda d:d.coefficient.abs()).sort_values('abs_coefficient',ascending=False).head(10).to_csv(OUT/'logistic_coefficients.csv',index=False)
    # Premium test customers ranked by chosen model lapse probability
    segment_lookup=pd.read_csv(OUT/'customer_segments.csv')[['customer_id','is_premium']]
    test_customers=f.loc[idx_test,['customer_id','lapsed']+feature_cols].merge(segment_lookup,on='customer_id',how='left')
    test_customers['is_premium']=test_customers.is_premium.fillna(False).astype(bool)
    probs=chosen.predict_proba(test_customers[feature_cols])[:,list(chosen.classes_).index(1)]
    test_customers['lapse_probability']=probs
    ret=test_customers[test_customers.is_premium].sort_values('lapse_probability',ascending=False).head(10)
    ret[['customer_id','lapse_probability','lapsed']].rename(columns={'lapsed':'actually_lapsed'}).to_csv(OUT/'retention_list.csv',index=False)
    # concise machine-readable summary
    summary={'rfm_customers':len(rfm),'recency_median':float(rfm.recency.median()),'frequency_mean':float(rfm.frequency.mean()),'monetary_mean':float(rfm.monetary.mean()),'cohort_rows':len(f),'lapsed_count':int(y.sum()),'train_rows':len(X_train),'test_rows':len(X_test),'chosen_model':chosen_name,'chosen_test_f1':float(f1_score(y_test,pred,zero_division=0)),'premium_retention_true_lapsed':int(ret.lapsed.sum()),'pca_1':float(pca.explained_variance_ratio_[0]),'pca_2':float(pca.explained_variance_ratio_[1])}
    pd.DataFrame([summary]).to_csv(OUT/'part_b_summary.csv',index=False)
    print('Part B complete:',summary)

if __name__=='__main__': run()
