from pathlib import Path
import json, sqlite3, time, uuid, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import mock_llm
RAW=ROOT/'data'/'raw'; POL=ROOT/'data'/'policies'; OUT=ROOT/'outputs'; STATE=ROOT/'state'; DB=ROOT/'data'/'instamart.db'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
LEDGER=OUT/'refund_ledger.jsonl'

def load_chunks(overlap=10):
    chunks=[]; size=40; step=size-overlap
    for path in sorted(POL.glob('*.txt')):
        words=path.read_text(encoding='utf-8').split()
        if not words: continue
        start=0
        while start<len(words):
            end=min(start+size,len(words))
            chunks.append({'source':path.name,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start+=step
    return chunks

CHUNKS=load_chunks(10)
VECTORIZER=TfidfVectorizer().fit([c['text'] for c in CHUNKS])
def retrieve(query,k=3,chunks=None,vectorizer=None):
    chunks=CHUNKS if chunks is None else chunks
    vectorizer=VECTORIZER if vectorizer is None else vectorizer
    mat=vectorizer.transform([c['text'] for c in chunks]); q=vectorizer.transform([query])
    scores=cosine_similarity(q,mat)[0]
    order=sorted(range(len(chunks)),key=lambda i:(-scores[i],i))[:k]
    return [{'source':chunks[i]['source'],'score':float(scores[i]),'text':chunks[i]['text']} for i in order]

def answer(question,chunks=None,vectorizer=None):
    hits=retrieve(question,3,chunks,vectorizer)
    context='CONTEXT:\n'+'\n'.join(f"[{h['source']}]\n{h['text']}" for h in hits)+'\nQUESTION:\n'+question
    messages=[{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present, say exactly that you could not find the information in the policy documents.'},{'role':'user','content':context}]
    response=mock_llm.chat(messages)
    return response.choices[0].message.content,[h['source'] for h in hits]

def policy_eval():
    qa=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False)
    rows=[]
    for overlap in [10,0]:
        chunks=load_chunks(overlap)
        vec=TfidfVectorizer().fit([c['text'] for c in chunks])
        for _,r in qa.iterrows():
            hits=retrieve(r.question,3,chunks,vec)
            ctx='CONTEXT:\n'+'\n'.join(f"[{h['source']}]\n{h['text']}" for h in hits)+'\nQUESTION:\n'+r.question
            response=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: answer only from supplied context and fallback if absent.'},{'role':'user','content':ctx}])
            txt=response.choices[0].message.content or ''
            rows.append({'overlap':overlap,'question':r.question,'expected_doc':r.expected_doc,'expected_fact':r.expected_fact,'top3_sources':'|'.join(h['source'] for h in hits),'doc_hit':bool(r.expected_doc!='NONE' and r.expected_doc in [h['source'] for h in hits]),'fact_hit':bool(r.expected_fact and r.expected_fact in txt),'fallback':txt.startswith('I could not find'),'answer':txt})
    pd.DataFrame(rows).to_csv(OUT/'policy_eval_results.csv',index=False)
    return rows

TOOLS_SPEC=[
 {'name':'get_order','description':'Look up a single order in the clean SQLite order table.','parameters':{'order_id':'string'}},
 {'name':'search_policy','description':'Retrieve the most relevant policy chunks for a query.','parameters':{'query':'string'}},
 {'name':'calculate_refund','description':'Calculate refund eligibility and amount from order amount and delivery time.','parameters':{'order_amount':'number','delivery_minutes':'number|null'}},
 {'name':'issue_refund','description':'Append an approved refund action to the refund ledger.','parameters':{'order_id':'string','refund_inr':'number'}},
 {'name':'escalate_to_human','description':'Append a refund escalation action to the refund ledger.','parameters':{'order_id':'string','refund_inr':'number'}}]

def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=? LIMIT 1',(order_id,)).fetchone()
    if row is None: return {'error':'order not found'}
    return {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':None if row[3] is None else int(row[3])}
def search_policy(query): return {'results':retrieve(query,k=3)}
def calculate_refund(order_amount,delivery_minutes):
    refund=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {'refund_inr':round(refund,2)}
def _ledger(tool_name,order_id,refund_inr):
    row={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with LEDGER.open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

class SimulatedCrash(RuntimeError): pass

def _save_checkpoint(run_id,scenario_id,messages,tokens_used,llm_calls,steps,tools_executed,iteration):
    obj={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'llm_calls':llm_calls,'step_number':iteration,'steps':steps,'tools_executed':tools_executed}
    (STATE/f'{run_id}.json').write_text(json.dumps(obj,default=str,indent=2),encoding='utf-8')

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None,resume=False):
    run_id=run_id or f'{scenario_id}_{uuid.uuid4().hex[:8]}'
    if resume:
        cp=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
        messages=cp['messages']; tokens_used=cp['tokens_used']; llm_calls=cp['llm_calls']; steps=cp['steps']; tools_executed=cp['tools_executed']; start_iter=cp['step_number']
        started=time.monotonic()
    else:
        messages=[{'role':'system','content':'AGENT: You are a careful refund workflow agent. Use tools to check orders and policy before deciding.'},{'role':'user','content':user_message}]
        tokens_used=0; llm_calls=0; steps=[]; tools_executed=[]; start_iter=0; started=time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic()-started >= time_limit_seconds:
            return {'scenario_id':scenario_id,'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        response=mock_llm.chat(messages); llm_calls+=1; tokens_used+=int(response.usage.total_tokens)
        if tokens_used>token_limit:
            return {'scenario_id':scenario_id,'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token budget exceeded.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        msg=response.choices[0].message; tool_calls=getattr(msg,'tool_calls',[]) or []
        if not tool_calls:
            final=msg.content or ''
            status=final.split(':',1)[0] if ':' in final else 'COMPLETED'
            return {'scenario_id':scenario_id,'status':status,'final_answer':final,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}
        tc_dicts=[]
        for tc in tool_calls:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            tc_dicts.append({'id':tc.id,'name':tc.name,'arguments':args})
        messages.append({'role':'assistant','content':None,'tool_calls':tc_dicts})
        for tc in tc_dicts:
            observation=TOOL_MAP[tc['name']](**tc['arguments'])
            tools_executed.append(tc['name'])
            step={'step':len(steps)+1,'tool':tc['name'],'arguments':tc['arguments'],'observation':observation}
            steps.append(step)
            messages.append({'role':'tool','tool_call_id':tc['id'],'name':tc['name'],'content':json.dumps(observation,sort_keys=True)})
            _save_checkpoint(run_id,scenario_id,messages,tokens_used,llm_calls,steps,tools_executed,llm_calls)
            if crash_after_tools is not None and len(tools_executed)>=crash_after_tools:
                raise SimulatedCrash(f'simulated crash after {crash_after_tools} tool executions')
    return {'scenario_id':scenario_id,'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; maximum model-call iterations reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed,'run_id':run_id}

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    cp=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    # The same mock call counter is deterministic from messages; reset is safe because its AGENT path is stateless.
    return run_agent('',cp['scenario_id'],max_iterations,token_limit,time_limit_seconds,run_id=run_id,resume=True)

def run():
    OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    for old_checkpoint in STATE.glob('*.json'):
        old_checkpoint.unlink()
    if LEDGER.exists(): LEDGER.unlink()
    # retrieval and grounding evaluations
    eval_rows=policy_eval()
    # query retrieval demo and chunks manifest
    pd.DataFrame(CHUNKS).to_csv(OUT/'policy_chunks.csv',index=False)
    top=retrieve('refund policy late delivery',3)
    pd.DataFrame(top).to_csv(OUT/'retrieval_demo.csv',index=False)
    # Agent scenarios
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False)
    agent_rows=[]
    resumed=None
    for _,s in scenarios.iterrows():
        if s['scenario_id']=='S1':
            # The S1 scenario itself is run with the required crash-and-resume demonstration.
            crash_id='S1_CRASH_RESUME'
            try:
                run_agent(s['message'],'S1',run_id=crash_id,crash_after_tools=2)
            except SimulatedCrash:
                pass
            result=resume_agent(crash_id)
            resumed=result
        else:
            result=run_agent(s['message'],s['scenario_id'],run_id=f"{s['scenario_id']}_RUN")
        result['expected_outcome']=s['expected_outcome']; result['expected_refund_inr']=float(s['expected_refund_inr'])
        agent_rows.append(result)
    with (OUT/'agent_runs.jsonl').open('w',encoding='utf-8') as f:
        for r in agent_rows:
            # include every tool's name, arguments and observation through steps
            f.write(json.dumps({k:r[k] for k in ['scenario_id','status','final_answer','llm_calls','tokens_used','steps','tools_executed','expected_outcome','expected_refund_inr']},default=str)+'\n')
    # Stop-safely demos. Ensure independent mock state is fine; agent behaviour is message-driven.
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1['message'],'S1_TOKEN_DEMO',token_limit=200,run_id='S1_TOKEN_DEMO')
    timeout_demo=run_agent(s1['message'],'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='S1_TIMEOUT_DEMO')
    # S1 was already crash-resumed above, so its refund is not issued twice.
    demos={'token_limit_demo':token_demo,'timeout_demo':timeout_demo,'crash_resume':resumed,'crash_tools':resumed['tools_executed']}
    (OUT/'agent_demos.json').write_text(json.dumps(demos,indent=2,default=str),encoding='utf-8')
    # Policy evaluation summary split by overlap
    edf=pd.DataFrame(eval_rows)
    summaries=[]
    for overlap,g in edf.groupby('overlap'):
        answerable=g[g.expected_doc!='NONE']; unanswerable=g[g.expected_doc=='NONE']
        summaries.append({'overlap':overlap,'chunks':len(load_chunks(overlap)),'answerable_doc_hit':f"{int(answerable.doc_hit.sum())}/{len(answerable)}",'answerable_fact_hit':f"{int(answerable.fact_hit.sum())}/{len(answerable)}",'unanswerable_fallback':f"{int(unanswerable.fallback.sum())}/{len(unanswerable)}"})
    pd.DataFrame(summaries).to_csv(OUT/'policy_eval_summary.csv',index=False)
    print('Part D complete; chunks=',len(CHUNKS),'top sources=',[x['source'] for x in top])
    print('Agent scenarios:',[(r['scenario_id'],r['status'],r['llm_calls'],r['tokens_used']) for r in agent_rows])
    print('Demos:',{k:(v.get('status') if isinstance(v,dict) else v) for k,v in demos.items()})

if __name__=='__main__': run()
