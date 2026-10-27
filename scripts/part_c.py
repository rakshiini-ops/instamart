from pathlib import Path
import re, json, random, time
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
OUT=ROOT/'outputs'; RAW=ROOT/'data'/'raw'; OUT.mkdir(exist_ok=True)
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE=re.compile(r'\b[6-9]\d{9}\b')
ORDER=re.compile(r'ORD-\d{6}')
ALLOWED={'billing','delivery','technical','fraud'}; MAX_INPUT_TOKENS=400; BASE_DELAY=.05
ZERO_PROMPT='''You are a CLASSIFY assistant. Classify the supplied support ticket into exactly one allowed category: billing, delivery, technical, fraud. Return JSON only with this exact shape: {"category": "...", "reason": "..."}. Never include any other text.'''
FEW_PROMPT='''You are a CLASSIFY assistant. Classify the supplied support ticket into exactly one allowed category: billing, delivery, technical, fraud. Return JSON only with this exact shape: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: An unauthorized transaction was made on my card. Reply: {"category":"fraud","reason":"unauthorized transaction"}
Ticket: My order arrived late and the rider is unreachable. Reply: {"category":"delivery","reason":"late delivery"}
Ticket: I was charged twice for one order. Reply: {"category":"billing","reason":"duplicate charge"}'''

def redact(text):
    counts={}
    text,n=EMAIL.subn('[EMAIL]',text); counts['emails']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phones']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_ids']=n
    return text,counts

def parse_reply(text):
    try:
        obj=json.loads(text)
        cat=obj.get('category')
        if cat in ALLOWED: return cat,'classified_json'
        return '', 'flagged_invalid_output'
    except (ValueError,AttributeError,TypeError):
        words=re.findall(r'[a-z]+',(text or '').lower())
        found=[w for w in words if w in ALLOWED]
        if len(set(found))==1: return found[0],'classified_keyword'
        return '', 'flagged_invalid_output'

def call_with_retry(messages,ticket_id,logs):
    for retry_no in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no>=3: raise
            wait=BASE_DELAY*(2**retry_no)*random.uniform(.9,1.15)
            logs.append(f'retry ticket={ticket_id} retry={retry_no} wait={wait:.4f}s')
            print(logs[-1]); time.sleep(wait)
    raise RuntimeError('unreachable')

def run_pipeline(prompt, budget=None, save_path=None):
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    mock_llm.reset(); random.seed(42)
    total_tokens=0; rows=[]; retry_logs=[]; redaction_totals={'emails':0,'phones':0,'order_ids':0}; calls=0
    for _,r in tickets.iterrows():
        raw=str(r['text']); ticket_id=r['ticket_id']; red,counts=redact(raw)
        for k,v in counts.items(): redaction_totals[k]+=v
        assert not EMAIL.search(red) and not PHONE.search(red) and not ORDER.search(red), f'PII remained in {ticket_id}'
        base={'ticket_id':ticket_id,'customer_id':r['customer_id'],'true_category':r['true_category'],'redacted_text':red,'category':'','status':'','tokens_used':0}
        if not raw.strip(): base['status']='skipped_empty'; rows.append(base); continue
        if mock_llm.est_tokens(raw)>MAX_INPUT_TOKENS: base['status']='flagged_long'; rows.append(base); continue
        if budget is not None and total_tokens>=budget:
            base['status']='budget_exhausted'; rows.append(base); continue
        messages=[{'role':'system','content':prompt},{'role':'user','content':red}]
        # Strict privacy assertion applies to the actual user message
        assert messages[-1]['content']==red and not EMAIL.search(messages[-1]['content']) and not PHONE.search(messages[-1]['content']) and not ORDER.search(messages[-1]['content'])
        response=call_with_retry(messages,ticket_id,retry_logs); calls+=1
        used=response.usage.total_tokens; total_tokens+=used; base['tokens_used']=used
        cat,status=parse_reply(response.choices[0].message.content)
        base['category']=cat; base['status']=status
        rows.append(base)
    df=pd.DataFrame(rows)
    df.attrs.update(total_tokens=total_tokens,calls=calls,retry_logs=retry_logs,redaction_totals=redaction_totals)
    if save_path: df.to_csv(save_path,index=False)
    return df

def evaluate(df):
    classified=df[df.status.isin(['classified_json','classified_keyword'])]
    acc=(classified.category==classified.true_category).mean() if len(classified) else 0
    counts=df.status.value_counts().to_dict()
    return {'total_tickets':len(df),'classified':len(classified),'accuracy':acc,'total_tokens':int(df.attrs.get('total_tokens',df.tokens_used.sum())),'tokens_per_classified':float(df.attrs.get('total_tokens',df.tokens_used.sum())/max(len(classified),1)),'status_counts':counts}

def run():
    OUT.mkdir(exist_ok=True)
    zero=run_pipeline(ZERO_PROMPT)
    few=run_pipeline(FEW_PROMPT)
    # Confusion table for zero-shot, over classified rows
    z=zero[zero.status.isin(['classified_json','classified_keyword'])]
    pd.crosstab(z.true_category,z.category,dropna=False).to_csv(OUT/'triage_zero_shot_confusion.csv')
    pd.DataFrame([{'prompt':'zero_shot',**evaluate(zero)},{'prompt':'few_shot',**evaluate(few)}]).to_csv(OUT/'triage_prompt_comparison.csv',index=False)
    pd.DataFrame([{'type':k,'substitutions':v} for k,v in zero.attrs['redaction_totals'].items()]+[{'type':'total','substitutions':sum(zero.attrs['redaction_totals'].values())}]).to_csv(OUT/'redaction_counts.csv',index=False)
    # Budget demo: 6000 as required; if none exhausted, rerun at 5000
    budget_demo=run_pipeline(ZERO_PROMPT,budget=6000)
    budget_used=6000
    if not (budget_demo.status=='budget_exhausted').any():
        budget_used=5000; budget_demo=run_pipeline(ZERO_PROMPT,budget=budget_used)
    budget_demo.to_csv(OUT/'triage_budget_demo.csv',index=False)
    # Final few-shot pipeline; map customer to segment and route
    final=run_pipeline(FEW_PROMPT,budget=100000)
    seg=pd.read_csv(OUT/'customer_segments.csv') if (OUT/'customer_segments.csv').exists() else pd.DataFrame(columns=['customer_id','segment_name','is_premium'])
    seg['customer_id']=seg.customer_id.astype(str); lookup=seg.set_index('customer_id').to_dict('index')
    routes=[]
    for _,r in final.iterrows():
        s=lookup.get(str(r.customer_id),{}); premium=bool(s.get('is_premium',False)); segment=s.get('segment_name','') if s else ''
        if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route,priority='human_review','NORMAL'
        elif r.category=='fraud': route,priority='security_team','HIGH'
        elif premium: route,priority='premium_desk','HIGH'
        else: route,priority='support_queue','NORMAL'
        routes.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':segment,'priority':priority,'route':route,'tokens_used':r.tokens_used})
    pd.DataFrame(routes).to_csv(OUT/'triage_results.csv',index=False)
    # Save retry log and output summary
    (OUT/'triage_retry.log').write_text('\n'.join(final.attrs['retry_logs'])+'\n',encoding='utf-8')
    summary={'zero_shot':evaluate(zero),'few_shot':evaluate(few),'budget_demo':evaluate(budget_demo),'budget_used':budget_used,'budget_exhausted':int((budget_demo.status=='budget_exhausted').sum()),'redactions':zero.attrs['redaction_totals'],'redaction_total':sum(zero.attrs['redaction_totals'].values())}
    pd.DataFrame([{'summary_json':json.dumps(summary)}]).to_csv(OUT/'part_c_summary.csv',index=False)
    print('Part C complete:',json.dumps(summary))

if __name__=='__main__': run()
