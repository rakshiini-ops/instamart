# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part capstone using deterministic, fictional data and the supplied offline `mock_llm.py`. It combines a cleaned analytical database, customer segmentation and lapse prediction, privacy-safe ticket triage, policy retrieval, and a resumable refund agent. All monetary values are in Indian Rupees (INR). No API key or network model call is used.

The supplied `generate_data.py` and `mock_llm.py` are kept unmodified. Run the scripts in order because Parts B–D consume outputs from earlier parts.

## Reproduce

From the repository root, run these commands in order. On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1`; on macOS/Linux use `source .venv/bin/activate`.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python scripts/part_a.py`
7. `python scripts/part_b.py`
8. `python scripts/part_c.py`
9. `python scripts/part_d.py`
10. `python validate.py`

The last command prints `ALL CHECKPOINTS PASSED` when the required data checkpoints and output files are present. `python run_all.py` is a convenience command that runs the same pipeline from data generation through validation. Random seeds are fixed in the generator and model code. `tickets.csv` is read with `keep_default_na=False` in the ticket pipeline so empty ticket text remains an empty string.

## Part A — Data Foundation: Clean, Store, Query

### Raw data audit

The generated raw table shapes are: customers **1,200 × 5**, products **40 × 4**, orders **13,084 × 9**, and tickets **164 × 5**. In the table below, a blank string is counted as missing for customer/order/ticket text columns. Full audit output is saved to `outputs/raw_audit.csv` and the requested headline counts to `outputs/raw_audit_metrics.csv`.

| Table | Column | Raw dtype | Missing / blank count | Missing % |
|---|---|---:|---:|---:|
| customers | customer_id | object | 0 | 0.00 |
| customers | city | object | 0 | 0.00 |
| customers | signup_date | object | 0 | 0.00 |
| customers | age | object | 61 | 5.08 |
| customers | gender | object | 30 | 2.50 |
| products | product_id | object | 0 | 0.00 |
| products | product_name | object | 0 | 0.00 |
| products | category | object | 0 | 0.00 |
| products | price_inr | int64 | 0 | 0.00 |
| orders | order_id | object | 0 | 0.00 |
| orders | customer_id | object | 0 | 0.00 |
| orders | product_id | object | 0 | 0.00 |
| orders | quantity | int64 | 0 | 0.00 |
| orders | order_ts | object | 0 | 0.00 |
| orders | status | object | 0 | 0.00 |
| orders | discount_pct | int64 | 0 | 0.00 |
| orders | order_amount | object | 0 | 0.00 |
| orders | delivery_minutes | object | 2,541 | 19.42 |
| tickets | ticket_id | object | 0 | 0.00 |
| tickets | customer_id | object | 0 | 0.00 |
| tickets | order_id | object | 8 | 4.88 |
| tickets | text | object | 4 | 2.44 |
| tickets | true_category | object | 0 | 0.00 |

| Raw audit metric | Count |
|---|---:|
| Order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with `pending` status | 656 |
| Distinct `order_id` values | 12,309 |
| Unparseable `order_amount` entries | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Blank customer ages | 61 |
| Non-blank ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and validation

- **Customers:** trim `city` and apply title case, changing 476 city strings and producing six cities. Normalize gender variants to `male`, `female`, or `unknown`; 950 gender values change during normalization, including 30 blanks mapped to `unknown`. The 61 blank ages and 6 out-of-range ages (67 values total) are filled using the median age of the customer's city with `groupby(...).transform('median')`. Convert all 1,200 `signup_date` values to dates. The six-city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184. Clean gender counts are female 610, male 560 and unknown 30. There are no missing ages; ages range from 18 to 60.
- **Orders:** parse all `order_ts` values; remove 119 exact duplicate rows; then remove 656 older records and keep the latest timestamp for each `order_id` (one row per order). After this deduplication, 112 remaining unparseable `order_amount` values are filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. Of the 106 raw negative delivery times, 105 remain after latest-record deduplication and are set to missing. After this rule, 774 missing delivery times on completed orders are filled with the median for that product category among completed orders, using `groupby` and `transform`; all 1,199 cancelled orders retain missing delivery time because they were not delivered. The cleaned table has 12,309 rows: 11,110 completed and 1,199 cancelled, no pending rows, no missing amounts, no missing completed delivery times, and 1,199 cancelled rows with missing delivery time. Completed revenue is **INR 2,998,668.80**.
- Clean CSVs are in `data/clean/`; SQLite tables `customers`, `products` and `orders` are in `data/instamart.db`. `order_ts` is stored as `YYYY-MM-DD HH:MM:SS` text. The validation script asserts more than eight reference checkpoints.

### SQL analysis

All seven queries are saved in `sql/queries.sql`, executed against SQLite with `pd.read_sql_query`, and their result tables are saved as `outputs/query_q1.csv` through `outputs/query_q7.csv`.

| Query | Result |
|---|---|
| Q1 — `WHERE` and `HAVING` | 30 customers have completed orders over INR 500 whose grouped total exceeds INR 8,000. |
| Q2 — three-table join | 30 city/category cells; total revenue INR 2,998,668.80. Largest cell: Bengaluru / Electronics, INR 233,909.45. |
| Q3 — scalar subquery | 11 products cost more than the average catalogue price. |
| Q4 — `LEFT JOIN` | 25 customers have no completed order. |
| Q5 — CTEs | 550 customers are above average completed spend; average spend per customer with completed spend is INR 2,552.06. |
| Q6 — ranking windows | Top customer per city: Bengaluru C0082 (INR 12,856.20); Chennai C0930 (INR 10,151.40); Delhi C0291 (INR 13,742.60); Hyderabad C0743 (INR 12,730.05); Mumbai C0693 (INR 14,766.55); Pune C0426 (INR 10,377.10). There are 8 rows where `RANK()` differs from `ROW_NUMBER()`: ties share a rank, while row numbers remain unique. |
| Q7 — running total | 183 daily rows; the final cumulative revenue is INR 2,998,668.80. |

### pandas cross-check and descriptive statistics

The city/category `pivot_table(..., margins=True)` grand total equals the SQL total, INR 2,998,668.80. Completed-order city aggregates include Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894; and Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613. All city aggregates are in `outputs/city_spend_stats.csv`.

| Completed `order_amount` statistic | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | INR 64.80 / INR 297.00 |
| IQR fences | Lower INR -283.50; upper INR 645.30 |
| Outliers below / above fences | 0 / 897 |

The orders table describes synthetic completed and cancelled grocery, dairy, snacks, personal-care and electronics purchases. Amounts are strongly right-skewed: the mean (INR 269.91) is much higher than the median (INR 135.00), with 897 upper-fence outliers, partly reflecting expensive electronics. The largest absolute correlation is order amount with discount percentage (about −0.095); order amount and quantity are about +0.087. Both relationships are weak, and correlation of this size does not establish causation. Negative and blank delivery times and non-numeric raw amounts were anomalies handled by the documented cleaning rules.

## Part B — Customer Segments and Lapse Prediction

### RFM and K-means

The whole-window RFM table uses completed orders through the snapshot `2025-10-01 00:00:00`. It contains 1,175 customers; recency median 19 days (minimum 0, maximum 180), frequency mean 9.46 (maximum 33), and monetary mean INR 2,552.06 (maximum INR 14,766.55). RFM values are in `outputs/rfm_table.csv`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

I chose **K=4**: it has the best silhouette score among the tested K values and a clear business interpretation. Inertia always decreases as K increases because additional clusters can only make within-cluster fit at least as good; K=8's lower inertia alone does not justify eight harder-to-action customer groups. K-means was fit on standardized RFM columns with `n_init=10` and `random_state=42`.

| Segment ID | Name | Size | Mean recency (days) | Mean frequency | Mean monetary | Premium? | Recommended action |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | At-Risk Occasional | 436 | 28.86 | 3.77 | INR 1,057.62 | No | Send a targeted win-back offer and ask for feedback. |
| 1 | Premium Loyalists | 138 | 24.12 | 6.77 | INR 7,129.70 | Yes | Protect loyalty with early access and tailored retention offers. |
| 2 | Frequent Regulars | 419 | 8.77 | 19.40 | INR 3,410.02 | No | Use replenishment reminders and loyalty rewards. |
| 3 | Dormant Customers | 182 | 109.04 | 2.20 | INR 685.99 | No | Run a reactivation campaign with a time-limited return offer. |

The Premium flag belongs to the segment with the highest mean monetary value. Cluster IDs are arbitrary labels, not an ordinal ranking. PCA on the scaled RFM matrix explains 0.6343 with PC1 and 0.2180 with PC2, 0.8523 combined (`outputs/pca_variance.csv`). The required projection is not committed as an image because the submission is text and code only.

### Lapse definition and evaluation

The cohort contains 1,156 customers with at least one completed order before `2025-09-01`; 443 (38.32%) lapsed because they had no completed order during 1–30 September. The stratified split is 924 training and 232 test rows. Features use only pre-cutoff orders; the task-3 segment labels are not model features. The `ColumnTransformer`/`Pipeline` fits the one-hot encoder and numeric scaler on training rows only, and the code asserts that the feature matrix has no missing values. No order from 2025-09-01 onward entered model features.

| Model | Best parameters | Best CV F1 | Test accuracy | Test precision (lapsed) | Test recall (lapsed) | Test F1 (lapsed) | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---|---:|---:|---:|---:|---:|---|
| Always predict not lapsed | — | — | 0.616 | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| Logistic regression | `C=100` | 0.651 | 0.728 | 0.644 | 0.652 | 0.648 | `[[111, 32], [31, 58]]` |
| Random forest | `max_depth=8, min_samples_leaf=5` | 0.692 | 0.690 | 0.591 | 0.618 | 0.604 | `[[105, 38], [34, 55]]` |

The selected persisted model is logistic regression because it is simpler to explain and has the stronger measured test F1 here (0.648 versus 0.604). Missing a customer who is about to lapse loses an opportunity to retain future spend, while wrongly targeting a loyal customer wastes a discount and may train them to wait for offers; the team should use the probability as a prioritisation aid rather than an automatic discount decision. The model is saved to `outputs/lapse_model.joblib`, reloaded, and its test predictions are asserted to match exactly.

| Training experiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression, unbalanced training set | 0.644 | 0.652 | 0.648 |
| Logistic regression, oversampled lapsed class | 0.589 | 0.854 | 0.697 |

Oversampling duplicates lapsed training rows only (seed 42); the test set remains unchanged. Recall moved most, rising by about 0.20, while precision fell modestly. This is useful when missing likely lapsers is costly, but it also means more loyal customers may receive outreach.

The five largest random-forest feature importances are frequency (0.269), monetary (0.240), recency (0.115), average delivery time (0.067) and cancellation rate (0.056). Example logistic coefficient readings: frequency is negative (−1.697), so higher frequency is associated with lower lapse odds in the fitted model; monetary is positive (+0.283), so monetary is positively associated with lapse odds conditional on the other features; a favourite category of Electronics is negative (−1.135) relative to the dropped reference category. These are model associations, not causal effects.

The top 10 test-set Premium customers by predicted lapse probability are saved in `outputs/retention_list.csv`; 3 of those 10 actually lapsed. The Premium flag is calculated from whole-window RFM, including the September window that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation.

## Part C — Support Ticket Triage with Guardrails

### Input safety, redaction and output validation

The pipeline processes tickets in source order. Four whitespace-empty tickets are marked `skipped_empty`; four tickets with estimated input length 734 tokens exceed `MAX_INPUT_TOKENS=400` and are marked `flagged_long`. Neither group reaches the model. Redaction is applied in the required order: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. The full run redacted 55 emails, 91 phone numbers and 132 order IDs, 278 substitutions total. Assertions confirm no matching email, phone or order ID remains in the redacted user prompt.

The zero-shot prompt contains no `EXAMPLES:` marker; the few-shot prompt has three labelled examples, including fraud. Both require JSON with one of `billing`, `delivery`, `technical`, `fraud`. Responses are accepted as `classified_json` only when valid JSON has an allowed category. For non-JSON text, lowercase alphabetic words are extracted with `re.findall(r"[a-z]+", text.lower())`; exactly one allowed category word is accepted as `classified_keyword`. Invalid output is never guessed and is routed to a human.

| Full-run result | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| Accuracy among classified tickets | 126/147 = 85.7% | 147/147 = 100.0% |
| Total estimated tokens | 11,584 | 18,136 |
| Tokens per classified ticket | 78.80 | 123.37 |
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |
| `skipped_empty` / `flagged_long` | 4 / 4 | 4 / 4 |

The few-shot prompt uses 6,552 more tokens across the batch (about 44.57 more per classified ticket, roughly 56.6% more) and improves accuracy by 14.3 percentage points, correcting the 21 zero-shot classification errors. In this deterministic evaluation the additional examples are worth the cost for a support workflow where sending a fraud report to ordinary billing support is consequential. The zero-shot confusion table is saved in `outputs/triage_zero_shot_confusion.csv`:

| True category | Predicted billing | Predicted delivery | Predicted fraud | Predicted technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The retry wrapper allows up to three retries with exponential backoff and jitter (`BASE_DELAY=0.05`). A full eligible run produces 19 transient errors, all recovered. The budget demo uses `TOKEN_BUDGET=6000`: 75 tickets are marked `budget_exhausted` because the budget is checked before each new call. The accumulated token count can exceed 6,000 on the call that takes it over the threshold; that response is still recorded, and the next ticket is stopped before a new call. `outputs/triage_budget_demo.csv` contains this run.

The final few-shot run uses a budget of 100,000 and produces 164 rows in `outputs/triage_results.csv`. Tickets skipped for empty text, flagged as long, invalid, or budget-exhausted go to `human_review` at NORMAL priority. Valid fraud tickets go to `security_team` at HIGH priority; otherwise Premium customers go to `premium_desk` at HIGH priority; the remaining tickets go to `support_queue` at NORMAL priority. A customer absent from the segment file is treated as not premium. Rule-validated classifications are routed automatically; invalid outputs and guardrail stops are sent to human review.

## Part D — Policy RAG Assistant and Refund Agent

### Retrieval and grounding

The five policy documents are split into 40-word chunks with 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2, **15 chunks total**. `TfidfVectorizer` fits the chunk text and cosine similarity ranks query matches. TF (term frequency) measures how often a term appears in a chunk; IDF reduces the weight of terms that appear in many chunks. TF-IDF is lexical, so it can miss semantically equivalent wording when the query and policy use different words. For `refund policy late delivery`, the top three retrieved chunks all come from `refund_policy.txt`.

| Evaluation | Answerable doc in top 3 | Answerable expected fact found | Unanswerable fallback |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, “What delivery time makes a completed order eligible for a full refund?” loses the exact phrase “more than 45 minutes” across the chunk boundary; the returned answer mentions a full refund but does not contain the expected fact substring. The payment-failure question also fails its expected-fact substring check. The policy assistant sends a system message marked `POLICY ASSISTANT` and a user message containing only the required `CONTEXT:` / `[source]` chunks / `QUESTION:` structure; unanswerable questions receive the `I could not find...` fallback.

### Refund tools and seven scenarios

`TOOLS_SPEC` describes the five tools and `TOOL_MAP` maps names to local Python functions. `get_order` reads `data/instamart.db`; `search_policy` reuses retrieval; `calculate_refund` returns the full amount only for completed orders delivered in more than 45 minutes; `issue_refund` and `escalate_to_human` append timestamped JSON lines to `outputs/refund_ledger.jsonl`. Refunds above INR 1,000 are escalated rather than issued. All scenarios are assumed to be inside the seven-day request window.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 947 |
| S2 | REFUND | 135.00 | 5 | 947 |
| S3 | ESCALATE | 1,999.00 (escalated, not auto-issued) | 5 | 934 |
| S4 | DENIED | 0.00 | 4 | 645 |
| S5 | DENIED | 0.00 | 3 | 367 |
| S6 | NOT_FOUND | 0.00 | 2 | 92 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,060 |

The run log `outputs/agent_runs.jsonl` records the status, final answer, model-call count, estimated tokens, and each tool step's name, arguments and observation. S7 completes six model calls but has no final answer, so it safely returns `FALLBACK_MAX_ITERATIONS` under the default token limit of 6,000.

### Stop-safely demonstrations

- S1 with `token_limit=200` returns **`FALLBACK_TOKEN_LIMIT`** after 384 accumulated tokens; the tool requested by the over-limit response is not executed.
- S1 with `time_limit_seconds=0` returns **`FALLBACK_TIMEOUT`** before the first model call (0 calls, 0 tokens).
- Crash-and-resume S1 finishes as **`REFUND: INR 128.25 issued.`** The simulated crash occurs after the second tool execution has been checkpointed. The complete tool sequence across the crash and resume is exactly: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; no tool is repeated. Checkpoint files are stored in `state/`.

### ReAct versus plan-and-execute (under 250 words)

The measured ReAct path used 5 model calls / 947 estimated tokens for S1, 3 calls / 367 tokens for S5, and 2 calls / 92 tokens for S6. ReAct spends additional calls because the model observes each tool result before choosing the next action; this is useful when an observation is unexpected. For example, S6 stops immediately after `get_order` returns `order not found`, rather than attempting a refund, and a cancelled order is denied after the order lookup. A plan-and-execute design could ask for the whole tool sequence once, usually reducing calls on predictable paths, then replan when a tool returns an unexpected result. Its full-plan response can be longer, so token savings are not guaranteed; an unexamined fixed plan is also riskier around `NOT_FOUND`, cancelled orders, or changed policy results. For this refund workflow, I prefer ReAct because policy compliance and safe handling of observations matter more than minimising a few local model calls. The plan-and-execute token counts were not separately measured in this implementation; the figures above are measured ReAct results.

## Limitations and Next Steps

All records and policies are synthetic and the mock model is deterministic; this is not a production customer-support system. TF-IDF does not understand semantic similarity, the refund policy is simplified, and the lapse model is trained on synthetic historical patterns. A production version would require access controls, audit retention, human approval workflows, monitoring, model drift checks and evaluation on approved real-world data. The Premium flag is a post-hoc whole-window label that includes September, so the retention list is illustrative and not a leakage-free evaluation. Do not use the retention scores or simulated refunds to make real customer decisions.
