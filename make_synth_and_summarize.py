# %%
# Synthetic AR Portfolio Generator → KPIs → Rules (Refactored)
# Save as: synthetic_ar_summary.py  (run: python synthetic_ar_summary.py)

import os, json
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

np.random.seed(42)

OUT_DIR = "./synthetic_ar"
os.makedirs(OUT_DIR, exist_ok=True)
today = date.today()

N_CUSTOMERS = 50
INV_DAYS_BACK = 120
CURRENCY = "USD"

# ============================================
# Helpers
# ============================================
def money(cents: int) -> str:
    return f"${cents/100:,.0f}"

# ============================================
# 0) Generate Data (same as before)
# ============================================
# Customers
countries = ["US", "IN", "GB", "DE", "SG", "AE"]
customers = pd.DataFrame({
    "customer_id": range(1, N_CUSTOMERS + 1),
    "name": [f"Customer {i:03d}" for i in range(1, N_CUSTOMERS + 1)],
    "billing_country": np.random.choice(countries, size=N_CUSTOMERS),
    "credit_limit_cents": np.random.randint(500_000, 5_000_000, size=N_CUSTOMERS),
    "risk_score": np.round(np.random.uniform(40, 90, size=N_CUSTOMERS), 1),
    "status": np.random.choice(["active","on_hold","closed"], p=[0.8,0.15,0.05], size=N_CUSTOMERS)
})

# Invoices
rows, inv_id = [], 1
for cid in customers["customer_id"]:
    for _ in range(np.random.randint(5, 21)):
        inv_date = today - timedelta(days=int(np.random.randint(0, INV_DAYS_BACK)))
        due_date = inv_date + timedelta(days=30)
        amt = int(np.random.randint(2_000, 50_000)) * 100
        pay_frac = float(np.clip(np.random.beta(1.5, 2.5), 0, 1))
        balance = int(round(amt * (1 - pay_frac)))
        status = "paid" if balance <= 0 else np.random.choice(
            ["open","partially_paid","disputed"], p=[0.5,0.35,0.15]
        )
        rows.append([inv_id, cid, f"INV-{cid:03d}-{inv_id:05d}", inv_date, due_date,
                     CURRENCY, amt, max(0, balance), status])
        inv_id += 1

invoices = pd.DataFrame(rows, columns=[
    "invoice_id","customer_id","invoice_number","invoice_date","due_date",
    "currency","amount_cents","balance_cents","status"
])

# Payments (linked + unapplied)
payments, pay_id = [], 1
for _, inv in invoices.iterrows():
    total_paid = int(inv["amount_cents"]) - int(inv["balance_cents"])
    if total_paid <= 0:
        continue
    num_p = np.random.randint(1, 4)
    remaining = total_paid
    for i in range(num_p):
        amt = remaining if i == num_p - 1 else int(remaining * float(np.random.uniform(0.2, 0.6)))
        if i != num_p - 1:
            remaining -= amt
        pay_date = inv["invoice_date"] + timedelta(days=int(np.random.randint(5, 70)))
        payments.append([
            pay_id, int(inv["invoice_id"]), int(inv["customer_id"]), pay_date, int(amt),
            np.random.choice(["ACH","Wire","Check","Card"]), f"REF-{pay_id:06d}", True
        ])
        pay_id += 1

# add unapplied
for cid in np.random.choice(customers["customer_id"], size=int(N_CUSTOMERS*0.4), replace=False):
    payments.append([
        pay_id, None, int(cid), today - timedelta(days=int(np.random.randint(0, 10))),
        int(np.random.randint(500, 5000))*100,
        np.random.choice(["ACH","Wire","Check","Card"]), f"REF-{pay_id:06d}", False
    ])
    pay_id += 1

payments = pd.DataFrame(payments, columns=[
    "payment_id","invoice_id","customer_id","payment_date","amount_cents",
    "method","reference_number","applied"
])

# Promises to pay
ptp_rows = []
for cid in np.random.choice(customers["customer_id"], size=N_CUSTOMERS, replace=True):
    for _ in range(np.random.randint(0, 4)):
        ptp_rows.append([
            len(ptp_rows)+1, int(cid),
            int(np.random.randint(1_000, 20_000))*100,
            today - timedelta(days=int(np.random.randint(0, 60))),
            np.random.choice(["open","kept","failed","cancelled"], p=[0.35,0.35,0.2,0.1]),
            np.random.choice(["call","email","meeting","other"]), 1
        ])
promises = pd.DataFrame(ptp_rows, columns=[
    "promise_id","customer_id","promised_amount_cents","promised_date","status","source","created_by"
])

# Disputes
eligible = invoices[invoices["balance_cents"] > 0].sample(frac=0.15, random_state=42)
disp_rows = []
for _, inv in eligible.iterrows():
    opened = inv["invoice_date"] + timedelta(days=int(np.random.randint(10, 80)))
    amount = int(int(inv["balance_cents"]) * float(np.random.uniform(0.3, 0.9)))
    status = np.random.choice(["open","in_review","resolved","rejected"], p=[0.35,0.35,0.2,0.1])
    resolved = opened + timedelta(days=int(np.random.randint(5, 30))) if status in ("resolved","rejected") else pd.NaT
    disp_rows.append([
        len(disp_rows)+1, int(inv["invoice_id"]), int(inv["customer_id"]),
        opened, amount,
        np.random.choice(["pricing error","damaged goods","quantity short","PO mismatch"]),
        status, "Synthetic", resolved
    ])
disputes = pd.DataFrame(disp_rows, columns=[
    "dispute_id","invoice_id","customer_id","opened_date","amount_cents",
    "reason","status","resolution_note","resolved_date"
])

# Save CSVs (optional)
for name, df in {
    "customers": customers, "invoices": invoices, "payments": payments,
    "promises_to_pay": promises, "disputes": disputes
}.items():
    df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
print("✅ CSVs saved to", OUT_DIR)

# ============================================
# 1) KPI computation (ALL math lives here)
# ============================================
def compute_kpis(customers, invoices, payments, promises, disputes, today: date):
    inv = invoices.copy()
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
    inv["due_date"] = pd.to_datetime(inv["due_date"])
    inv["days_past_due"] = (pd.to_datetime(today) - inv["due_date"]).dt.days.clip(lower=0)

    # Aging buckets & totals
    inv["aging_bucket"] = pd.cut(
        inv["days_past_due"],
        bins=[-1,0,30,60,999999],
        labels=["current","30+","60+","90+"],
        right=True
    )
    aging_series = (
        inv.groupby("aging_bucket", observed=True)["balance_cents"]
          .sum()
          .reindex(["30+","60+","90+"])
          .fillna(0)
          .astype(int)
    )
    aging = {k:int(v) for k,v in aging_series.items()}

    # Customer utilization
    cust_bal = inv[inv["status"].isin(["open","partially_paid","disputed"])] \
                 .groupby("customer_id", observed=True)["balance_cents"].sum() \
                 .reset_index(name="outstanding_cents")
    cust = customers.merge(cust_bal, on="customer_id", how="left").fillna({"outstanding_cents":0})
    cust["credit_utilization_pct"] = np.round(
        100 * cust["outstanding_cents"] / cust["credit_limit_cents"], 2
    )
    high_util = cust[cust["credit_utilization_pct"] >= 95.0] \
                  .sort_values(["credit_utilization_pct","outstanding_cents"], ascending=False)

    # DSO (90d heuristic)
    cutoff_date = today - timedelta(days=90)
    last_pay = payments.groupby("invoice_id")["payment_date"].max() if not payments.empty \
               else pd.Series(dtype="datetime64[ns]")
    inv2 = inv.join(last_pay.rename("effective_pay_date"), on="invoice_id")
    inv2["effective_pay_date"] = pd.to_datetime(inv2["effective_pay_date"]).fillna(pd.to_datetime(today))
    inv_win = inv2[inv2["invoice_date"].dt.date >= cutoff_date].copy()
    if inv_win["amount_cents"].sum() > 0:
        inv_win["days_to_pay_or_elapsed"] = (inv_win["effective_pay_date"] - inv_win["invoice_date"]).dt.days.clip(lower=0)
        dso_90 = float(round(
            (inv_win["days_to_pay_or_elapsed"] * inv_win["amount_cents"]).sum()
            / inv_win["amount_cents"].sum(),
            2
        ))
    else:
        dso_90 = None

    # Failed promises (30d)
    cutoff_30 = today - timedelta(days=30)
    failed_promises_30d = int(len(promises[(promises["status"]=="failed") & (promises["promised_date"] >= cutoff_30)]))

    # Unapplied cash (latest snapshot per customer)
    ua = payments[payments["applied"] == False]
    if not ua.empty:
        latest_unapplied = (ua.groupby(["customer_id","payment_date"], as_index=False)
                              .agg(amount_cents=("amount_cents","sum"))
                              .sort_values(["customer_id","payment_date"])
                              .drop_duplicates("customer_id", keep="last"))
        total_unapplied = int(latest_unapplied["amount_cents"].sum())
    else:
        total_unapplied = 0

    # NEW: Dispute rate (last 90d) = disputes opened / invoices issued, in %
    cutoff_90 = pd.Timestamp(today - timedelta(days=90))
    recent_invoices = inv[inv["invoice_date"] >= cutoff_90]
    disputes["opened_date"] = pd.to_datetime(disputes["opened_date"], errors="coerce")
    recent_disputes = disputes[disputes["opened_date"] >= cutoff_90]

    dispute_rate_90 = round((len(recent_disputes) / max(len(recent_invoices), 1)) * 100.0, 2)

    kpis = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dso_90": dso_90,
        "aging_30": aging.get("30+", 0),
        "aging_60": aging.get("60+", 0),
        "aging_90": aging.get("90+", 0),
        "high_util_accounts": high_util[["customer_id","name","credit_limit_cents",
                                         "outstanding_cents","credit_utilization_pct"]].to_dict("records"),
        "high_util_count": int(len(high_util)),
        "failed_promises_30d": failed_promises_30d,
        "unapplied_cents": total_unapplied,
        "dispute_rate_90": dispute_rate_90
    }
    return kpis

# ============================================
# 2) Rules application (ONLY logic/thresholds)
# ============================================
RULES_CFG = {
    "util_threshold": 95.0,
    "dso_warn": 55.0,
    "dispute_rate_warn": 12.0,   # example threshold for dispute rate
}

def apply_rules(k, cfg=RULES_CFG):
    rules, recos = [], []

    if k["aging_90"] > 0:
        rules.append(f"Significant 90+ aging: {money(k['aging_90'])}.")
        recos.append("Escalate collections on 90+ bucket; prioritize highest balances.")

    if k["high_util_count"] > 0:
        top = k["high_util_accounts"][0]
        rules.append(f"{k['high_util_count']} high-utilization accounts (top: {top['name']} at {top['credit_utilization_pct']}%).")
        recos.append(f"Review credit limits and consider soft holds for ≥{cfg['util_threshold']}% utilization.")

    if k["failed_promises_30d"] > 0:
        rules.append(f"{k['failed_promises_30d']} failed promises (30d).")
        recos.append("Tighten PTP follow-up cadence and require confirmations in writing.")

    if k["unapplied_cents"] > 0:
        rules.append(f"Unapplied cash pending: {money(k['unapplied_cents'])}.")
        recos.append("Apply unapplied cash within 48 hours to reduce reported DSO.")

    if k["dso_90"] and k["dso_90"] > cfg["dso_warn"]:
        recos.append("Investigate invoice-to-cash delays; consider early-payment incentives.")

    # new rule: dispute rate
    if k["dispute_rate_90"] > cfg["dispute_rate_warn"]:
        rules.append(f"High dispute rate (90d): {k['dispute_rate_90']}%.")
        recos.append("Audit root causes for disputes (pricing/quantity/PO mismatch).")

    if not recos:
        recos.append("Portfolio stable — continue standard monitoring.")

    return rules, recos

# ============================================
# Run pipeline
# ============================================
kpis = compute_kpis(customers, invoices, payments, promises, disputes, today)
rules, recos = apply_rules(kpis)

# Summaries
md = f"""# Portfolio Summary (Rule-Based)

- **DSO (90d):** {kpis['dso_90'] if kpis['dso_90'] is not None else "N/A"} days  
- **Aging (30+/60+/90+):** {money(kpis['aging_30'])} / {money(kpis['aging_60'])} / {money(kpis['aging_90'])}  
- **High Utilization (≥{RULES_CFG['util_threshold']}%)**: {kpis['high_util_count']} account(s)  
- **Failed Promises (30d):** {kpis['failed_promises_30d']}  
- **Unapplied Cash:** {money(kpis['unapplied_cents'])}  
- **Dispute Rate (90d):** {kpis['dispute_rate_90']}%

**Signals:** {' '.join(rules) if rules else 'None.'}

**Actions:** 
- {recos[0]}
{''.join([f"- {r}\\n" for r in recos[1:]])}
"""

summary = {
    "generated_at": kpis["generated_at"],
    "kpis": kpis,
    "rules_triggered": rules,
    "recommendations": recos,
    "narrative_markdown": md
}

# Save
json_path = os.path.join(OUT_DIR, "portfolio_summary.json")
md_path   = os.path.join(OUT_DIR, "portfolio_summary.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

print("\n== KPI snapshot ==")
print(json.dumps({
    "DSO_90d_days": kpis["dso_90"],
    "aging_30_cents": kpis["aging_30"],
    "aging_60_cents": kpis["aging_60"],
    "aging_90_cents": kpis["aging_90"],
    "high_util_count": kpis["high_util_count"],
    "failed_promises_30d": kpis["failed_promises_30d"],
    "unapplied_cash_total_cents": kpis["unapplied_cents"],
    "dispute_rate_90_pct": kpis["dispute_rate_90"],
}, indent=2))

print("\n✅ Summary JSON:", json_path)
print("✅ Summary Markdown:", md_path)
