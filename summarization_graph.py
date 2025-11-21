from __future__ import annotations
import os, uuid, json
from typing import TypedDict, List, Optional, Dict, Any
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from langgraph.graph import StateGraph, END
import requests  # optional; only used if WEBHOOK_URL is set


# -----------------------------
# Config & helpers
# -----------------------------
load_dotenv()
DATA_DIR = "./synthetic_ar"

def money(cents: int) -> str:
    return f"${cents/100:,.0f}"


# -----------------------------
# State definition
# -----------------------------
class GraphState(TypedDict, total=False):
    run_id: str
    user_id: str
    stage: str

    raw_customers: pd.DataFrame
    raw_invoices: pd.DataFrame
    raw_payments: pd.DataFrame
    raw_promises: pd.DataFrame
    raw_disputes: pd.DataFrame

    kpis: Dict[str, Any]
    summary: Dict[str, Any]          # Node D output (rule-based JSON)
    validated: Dict[str, Any]        # Node E output (validated & repaired)
    error: str


# -----------------------------
# Node A: login_trigger
# -----------------------------
def node_login_trigger(state: GraphState) -> GraphState:
    state = dict(state)
    state["run_id"] = state.get("run_id") or str(uuid.uuid4())
    state["stage"] = "login_trigger"
    return state


# -----------------------------
# Node B: data_retrieval (CSV only)
# -----------------------------
def _load_from_csv() -> Dict[str, pd.DataFrame]:
    def read_csv(name: str, parse_cols: Optional[List[str]] = None):
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Generate CSVs first.")
        df = pd.read_csv(path)
        if parse_cols:
            for c in parse_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    customers = read_csv("customers")
    invoices  = read_csv("invoices", ["invoice_date", "due_date"])
    payments  = read_csv("payments", ["payment_date"])
    promises  = read_csv("promises_to_pay", ["promised_date"])
    try:
        disputes = read_csv("disputes", ["opened_date", "resolved_date"])
    except FileNotFoundError:
        disputes = pd.DataFrame(columns=[
            "dispute_id", "invoice_id", "customer_id", "opened_date",
            "amount_cents", "reason", "status", "resolution_note", "resolved_date"
        ])

    return dict(
        customers=customers,
        invoices=invoices,
        payments=payments,
        promises=promises,
        disputes=disputes,
    )

def node_data_retrieval_csv(state: GraphState) -> GraphState:
    state = dict(state)
    state["stage"] = "data_retrieval_csv"
    try:
        data = _load_from_csv()
    except Exception as e:
        raise RuntimeError(f"retrieval_failed: {e}")

    state["raw_customers"] = data["customers"]
    state["raw_invoices"]  = data["invoices"]
    state["raw_payments"]  = data["payments"]
    state["raw_promises"]  = data["promises"]
    state["raw_disputes"]  = data["disputes"]
    return state


# -----------------------------
# Node C: metric_calculation
# -----------------------------
def compute_kpis(customers: pd.DataFrame,
                 invoices: pd.DataFrame,
                 payments: pd.DataFrame,
                 promises: pd.DataFrame,
                 disputes: pd.DataFrame,
                 today: date) -> Dict[str, Any]:
    inv = invoices.copy()
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"], errors="coerce")
    inv["due_date"]     = pd.to_datetime(inv["due_date"], errors="coerce")
    inv["days_past_due"] = (pd.Timestamp(today) - inv["due_date"]).dt.days.clip(lower=0)

    # Aging buckets
    inv["aging_bucket"] = pd.cut(
        inv["days_past_due"],
        bins=[-1, 0, 30, 60, 999999],
        labels=["current", "30+", "60+", "90+"],
        right=True,
    )
    aging_series = (
        inv.groupby("aging_bucket", observed=True)["balance_cents"]
           .sum().reindex(["30+", "60+", "90+"]).fillna(0).astype(int)
    )
    aging = {k: int(v) for k, v in aging_series.items()}

    # Utilization
    open_mask = inv["status"].isin(["open", "partially_paid", "disputed"])
    cust_bal = (
        inv[open_mask]
        .groupby("customer_id", observed=True)["balance_cents"]
        .sum()
        .reset_index(name="outstanding_cents")
    )
    cust = customers.merge(cust_bal, on="customer_id", how="left").fillna({"outstanding_cents": 0})
    cust["credit_utilization_pct"] = np.round(
        100 * cust["outstanding_cents"] / cust["credit_limit_cents"], 2
    )
    high_util = cust[cust["credit_utilization_pct"] >= 95.0].sort_values(
        ["credit_utilization_pct", "outstanding_cents"], ascending=False
    )

    # DSO (90d heuristic)
    cutoff_date = today - timedelta(days=90)
    last_pay = (
        payments.groupby("invoice_id")["payment_date"].max()
        if not payments.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    inv2 = inv.join(last_pay.rename("effective_pay_date"), on="invoice_id")
    inv2["effective_pay_date"] = pd.to_datetime(inv2["effective_pay_date"]).fillna(pd.Timestamp(today))
    inv_win = inv2[inv2["invoice_date"].dt.date >= cutoff_date].copy()
    if inv_win["amount_cents"].sum() > 0:
        inv_win["days_to_pay_or_elapsed"] = (
            inv_win["effective_pay_date"] - inv_win["invoice_date"]
        ).dt.days.clip(lower=0)
        dso_90 = float(
            round(
                (inv_win["days_to_pay_or_elapsed"] * inv_win["amount_cents"]).sum()
                / inv_win["amount_cents"].sum(),
                2,
            )
        )
    else:
        dso_90 = None

    # Failed promises 30d
    cutoff_30 = today - timedelta(days=30)
    promises["promised_date"] = pd.to_datetime(promises["promised_date"], errors="coerce")
    failed_promises_30d = int(
        len(
            promises[
                (promises["status"] == "failed")
                & (promises["promised_date"] >= pd.Timestamp(cutoff_30))
            ]
        )
    )

    # Unapplied cash
    ua = payments[payments["applied"] == False]
    if not ua.empty:
        latest_unapplied = (
            ua.groupby(["customer_id", "payment_date"], as_index=False)
              .agg(amount_cents=("amount_cents", "sum"))
              .sort_values(["customer_id", "payment_date"])
              .drop_duplicates("customer_id", keep="last")
        )
        total_unapplied = int(latest_unapplied["amount_cents"].sum())
    else:
        total_unapplied = 0

    # Dispute rate 90d
    disputes["opened_date"] = pd.to_datetime(disputes.get("opened_date"), errors="coerce")
    cutoff_90_ts = pd.Timestamp(today - timedelta(days=90))
    recent_invoices = inv[inv["invoice_date"] >= cutoff_90_ts]
    recent_disputes = (
        disputes[disputes["opened_date"] >= cutoff_90_ts]
        if not disputes.empty
        else pd.DataFrame()
    )
    dispute_rate_90 = round(
        (len(recent_disputes) / max(len(recent_invoices), 1)) * 100.0, 2
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dso_90": dso_90,
        "aging_30": aging.get("30+", 0),
        "aging_60": aging.get("60+", 0),
        "aging_90": aging.get("90+", 0),
        "high_util_accounts": high_util[
            ["customer_id", "name", "credit_limit_cents", "outstanding_cents", "credit_utilization_pct"]
        ].to_dict("records"),
        "high_util_count": int(len(high_util)),
        "failed_promises_30d": failed_promises_30d,
        "unapplied_cents": total_unapplied,
        "dispute_rate_90": dispute_rate_90,
    }

def node_metric_calculation(state: GraphState) -> GraphState:
    state = dict(state)
    state["stage"] = "metric_calculation"
    k = compute_kpis(
        customers=state["raw_customers"],
        invoices=state["raw_invoices"],
        payments=state["raw_payments"],
        promises=state["raw_promises"],
        disputes=state["raw_disputes"],
        today=date.today(),
    )
    state["kpis"] = k
    return state


# -----------------------------
# Node D: rule-based summarization (no LLM)
# -----------------------------
def _rule_based_summary(k: Dict[str, Any]) -> Dict[str, Any]:
    actions: List[str] = []
    rules_summary_parts: List[str] = []

    if k["aging_90"] > 0:
        rules_summary_parts.append(f"Significant 90+ aging: {money(k['aging_90'])}.")
        actions.append("Escalate collections on 90+; prioritize highest balances.")
    if k["high_util_count"] > 0:
        rules_summary_parts.append(f"{k['high_util_count']} high-utilization accounts ≥95%.")
        actions.append("Review credit limits; consider soft holds ≥95% utilization.")
    if k["failed_promises_30d"] > 0:
        rules_summary_parts.append(f"{k['failed_promises_30d']} failed promises in last 30 days.")
        actions.append("Tighten PTP follow-ups; require written confirmations.")
    if k["unapplied_cents"] > 0:
        rules_summary_parts.append(f"Unapplied cash: {money(k['unapplied_cents'])}.")
        actions.append("Apply unapplied cash within 48 hours to reduce reported DSO.")
    if k["dso_90"] and k["dso_90"] > 55:
        rules_summary_parts.append(f"Elevated DSO(90d): {k['dso_90']} days.")
        actions.append("Investigate invoice-to-cash delays; consider early-payment incentives.")
    if k["dispute_rate_90"] > 12:
        rules_summary_parts.append(f"High dispute rate 90d: {k['dispute_rate_90']}%.")
        actions.append("Audit dispute root causes (pricing/quantity/PO mismatch).")
    if not actions:
        actions.append("Portfolio stable — continue standard monitoring.")

    md = (
        f"# Portfolio Summary (Rule-Based)\n\n"
        f"- **DSO (90d):** {k['dso_90'] if k['dso_90'] is not None else 'N/A'} days\n"
        f"- **Aging 30/60/90:** {money(k['aging_30'])} / {money(k['aging_60'])} / {money(k['aging_90'])}\n"
        f"- **High-Util Accounts:** {k['high_util_count']}\n"
        f"- **Failed Promises (30d):** {k['failed_promises_30d']}\n"
        f"- **Unapplied Cash:** {money(k['unapplied_cents'])}\n"
        f"- **Dispute Rate (90d):** {k['dispute_rate_90']}%\n\n"
        f"**Signals:** {' '.join(rules_summary_parts) if rules_summary_parts else 'None.'}\n\n"
        f"**Actions:**\n- " + "\n- ".join(actions)
    )

    return {
        "dso_90": k["dso_90"],
        "aging_30": k["aging_30"],
        "aging_60": k["aging_60"],
        "aging_90": k["aging_90"],
        "high_util_count": k["high_util_count"],
        "failed_promises_30d": k["failed_promises_30d"],
        "unapplied_cents": k["unapplied_cents"],
        "dispute_rate_90": k["dispute_rate_90"],
        "rules_summary": " ".join(rules_summary_parts),
        "actions_bullets": actions,
        "narrative_markdown": md,
        "_provider": "rule_based",
    }

def node_rule_summarization(state: GraphState) -> GraphState:
    """
    Node D: purely rule-based summary. No LLM calls, no external cost.
    """
    state = dict(state)
    state["stage"] = "rule_summarization"
    k = state["kpis"]
    parsed = _rule_based_summary(k)
    state["summary"] = parsed
    return state


# -----------------------------
# Node E: pydantic_validation (no LLM fix)
# -----------------------------
class PortfolioSummary(BaseModel):
    dso_90: Optional[float] = Field(default=None, ge=0)
    aging_30: int = Field(ge=0)
    aging_60: int = Field(ge=0)
    aging_90: int = Field(ge=0)
    high_util_count: int = Field(ge=0)
    failed_promises_30d: int = Field(ge=0)
    unapplied_cents: int = Field(ge=0)
    dispute_rate_90: float = Field(ge=0)
    rules_summary: Optional[str] = ""
    actions_bullets: List[str] = Field(default_factory=list)
    narrative_markdown: str

    @field_validator("actions_bullets", mode="before")
    @classmethod
    def ensure_list_of_str(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip("-• ").strip() for s in v.splitlines() if s.strip()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

def _coerce_nonneg_int(x) -> int:
    try:
        v = int(float(x))
        return max(0, v)
    except Exception:
        return 0

def _coerce_nonneg_float(x) -> float:
    try:
        v = float(x)
        return max(0.0, v)
    except Exception:
        return 0.0

def _repair_summary(summary: Dict[str, Any], kpis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic repairs: fill missing fields from KPIs, coerce types, clamp to non-negative.
    """
    s = dict(summary)
    for key in [
        "dso_90", "aging_30", "aging_60", "aging_90",
        "high_util_count", "failed_promises_30d",
        "unapplied_cents", "dispute_rate_90",
    ]:
        if s.get(key) is None:
            s[key] = kpis.get(key)

    s["aging_30"] = _coerce_nonneg_int(s.get("aging_30", 0))
    s["aging_60"] = _coerce_nonneg_int(s.get("aging_60", 0))
    s["aging_90"] = _coerce_nonneg_int(s.get("aging_90", 0))
    s["high_util_count"] = _coerce_nonneg_int(s.get("high_util_count", 0))
    s["failed_promises_30d"] = _coerce_nonneg_int(s.get("failed_promises_30d", 0))
    s["unapplied_cents"] = _coerce_nonneg_int(s.get("unapplied_cents", 0))
    s["dispute_rate_90"] = _coerce_nonneg_float(s.get("dispute_rate_90", 0))
    if s.get("dso_90") is not None:
        s["dso_90"] = _coerce_nonneg_float(s.get("dso_90"))

    if not s.get("narrative_markdown"):
        s["narrative_markdown"] = "# Summary\n\nNo narrative provided."
    if "actions_bullets" not in s or s["actions_bullets"] is None:
        s["actions_bullets"] = []
    return s

def node_pydantic_validation(state: GraphState) -> GraphState:
    """
    Validate Node D output; if invalid:
      1) heuristic repair; validate again
      2) if still invalid, record error
    """
    state = dict(state)
    state["stage"] = "pydantic_validation"

    summary = state.get("summary", {}) or {}
    kpis = state.get("kpis", {}) or {}

    try:
        model = PortfolioSummary(**summary)
        state["validated"] = json.loads(model.model_dump_json())
        return state
    except ValidationError:
        repaired = _repair_summary(summary, kpis)
        try:
            model = PortfolioSummary(**repaired)
            state["validated"] = json.loads(model.model_dump_json())
            return state
        except ValidationError as e2:
            state["error"] = f"validation_failed: {e2}"
            return state


# -----------------------------
# Node F: persist_and_notify (files + optional webhook)
# -----------------------------
def _to_markdown(v: Dict[str, Any]) -> str:
    def m(c): return f"${c/100:,.0f}"
    return (
        f"# Portfolio Summary\n\n"
        f"- **DSO (90d):** {v.get('dso_90', 'N/A')} days\n"
        f"- **Aging 30/60/90:** {m(v.get('aging_30',0))} / {m(v.get('aging_60',0))} / {m(v.get('aging_90',0))}\n"
        f"- **High-Util Accounts:** {v.get('high_util_count',0)}\n"
        f"- **Failed Promises (30d):** {v.get('failed_promises_30d',0)}\n"
        f"- **Unapplied Cash:** {m(v.get('unapplied_cents',0))}\n"
        f"- **Dispute Rate (90d):** {v.get('dispute_rate_90',0)}%\n\n"
        f"**Actions:**\n- " + "\n- ".join(v.get("actions_bullets", []) or ["(none)"])
    )

def node_persist_and_notify(state: GraphState) -> GraphState:
    """
    Persist validated summary to disk (JSON/MD) and optionally send webhook.
    """
    state = dict(state)
    state["stage"] = "persist_and_notify"

    validated = state.get("validated") or {}
    if not validated:
        state["error"] = state.get("error") or "nothing_to_persist"
        return state

    # 1) Save to files
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(DATA_DIR, f"portfolio_summary_{ts}.json")
    md_path = os.path.join(DATA_DIR, f"portfolio_summary_{ts}.md")

    if os.getenv("SAVE_JSON", "1") == "1":
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2)
        state["json_path"] = json_path

    narrative_md = validated.get("narrative_markdown") or _to_markdown(validated)
    if os.getenv("SAVE_MD", "1") == "1":
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(narrative_md)
        state["md_path"] = md_path

    # 2) Optional webhook notification
    try:
        hook = os.getenv("WEBHOOK_URL")
        if hook:
            title = "AR Summary Ready"
            k = validated
            text = (
                f"*{title}*  (run_id `{state.get('run_id')}`)\n"
                f"• DSO(90d): {k.get('dso_90')}\n"
                f"• Aging30/60/90: {k.get('aging_30')}/{k.get('aging_60')}/{k.get('aging_90')}\n"
                f"• High-util: {k.get('high_util_count')} | Failed PTP(30d): {k.get('failed_promises_30d')}\n"
                f"• Unapplied: {k.get('unapplied_cents')} | Dispute90: {k.get('dispute_rate_90')}%\n"
                + (
                    f"• Files: `{state.get('json_path','')}` `{state.get('md_path','')}`"
                    if state.get("json_path") or state.get("md_path")
                    else ""
                )
            )
            requests.post(hook, json={"text": text}, timeout=5)
            state["notified"] = True
    except Exception as e:
        state["notify_error"] = str(e)

    return state


# -----------------------------
# Build the graph (A → B → C → D → E → F)
# -----------------------------
builder = StateGraph(GraphState)
builder.add_node("login_trigger",         node_login_trigger)         # A
builder.add_node("data_retrieval_csv",   node_data_retrieval_csv)    # B
builder.add_node("metric_calculation",   node_metric_calculation)    # C
builder.add_node("rule_summarization",   node_rule_summarization)    # D
builder.add_node("pydantic_validation",  node_pydantic_validation)   # E
builder.add_node("persist_and_notify",   node_persist_and_notify)    # F

builder.set_entry_point("login_trigger")
builder.add_edge("login_trigger", "data_retrieval_csv")
builder.add_edge("data_retrieval_csv", "metric_calculation")
builder.add_edge("metric_calculation", "rule_summarization")
builder.add_edge("rule_summarization", "pydantic_validation")
builder.add_edge("pydantic_validation", "persist_and_notify")
builder.add_edge("persist_and_notify", END)

graph = builder.compile()


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    initial: GraphState = {"user_id": "analyst_123"}
    out = graph.invoke(initial)
    print("== RUN COMPLETE ==")
    print("stage:", out.get("stage"))
    print("run_id:", out.get("run_id"))
    print("kpis:", json.dumps(out.get("kpis", {}), indent=2, default=str))
    print("summary_raw:", json.dumps(out.get("summary", {}), indent=2, default=str))
    print("validated:", json.dumps(out.get("validated", {}), indent=2, default=str))
    if out.get("error"):
        print("ERROR:", out["error"])
