# summarize_portfolio.py
import json, mysql.connector as mysql
from datetime import date, timedelta
from pydantic_schema import PortfolioSummary, validate_portfolio_summary
from textwrap import dedent


# ... after you get raw_json from the LLM:

# ---- DB connect ----
conn = mysql.connect(
    host="localhost", user="root", password="Astrokid1*", database="ar_summarization"
)
cur = conn.cursor(dictionary=True)

# ---- 1) Fetch KPIs ----
cur.execute("CALL kpi_aging_buckets()")
aging = {row["aging_bucket"]: int(row["total_overdue_cents"]) for row in cur.fetchall()}

cur.execute("CALL kpi_high_utilization(%s)", (90.00,))
high_util = [
    {
      "customer": r["name"],
      "util_pct": float(r["credit_utilization_pct"]),
      "outstanding_cents": int(r["outstanding_cents"])
    }
    for r in cur.fetchall()
]

cur.execute("SET @dso := NULL")
cur.execute("CALL kpi_portfolio_dso(%s, @dso)", (90,))
cur.execute("SELECT @dso AS dso")
dso = cur.fetchone()["dso"]
dso = float(dso) if dso is not None else None

cur.execute(
    "CALL kpi_failed_promises(%s,%s)",
    (date.today() - timedelta(days=30), date.today())
)
failed_promises_30d = len(cur.fetchall())

kpi_snapshot = {
    "dso_90d": dso,
    "aging": aging,
    "high_utilization": high_util,
    "failed_promises_30d": failed_promises_30d,
}

# ---- 2) Build the LLM prompt ----
system = "You are a precise AR analyst. Be concise, numeric, and actionable."
user = dedent(f"""
Write a 6–10 sentence portfolio summary for Accounts Receivable.

Data (JSON):
{kpi_snapshot}

Rules:
- Plain English, bullet points allowed.
- Call out top risks, especially 90+ aging and high utilization > 90%.
- Include a one-line action recommendation.
- Keep it under 130 words.
- Do not invent numbers not in the JSON.
- Output JSON in this exact schema keys only:
  {{ "dso_days_90d": float|null,
     "aging_overview": {{"30+": int, "60+": int, "90+": int}},
     "high_utilization_accounts": [{{"customer": str, "util_pct": float, "outstanding_cents": int}}],
     "failed_promises_30d": int,
     "key_risks": [{{"name": str, "reason": str, "severity": "low"|"medium"|"high"}}],
     "notes": str|null }}
Return ONLY JSON.
""").strip()

# ---- 3) Call your LLM (example: OpenAI; replace with Bedrock if you use that) ----
from openai import OpenAI
client = OpenAI()  # assumes OPENAI_API_KEY env var

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"system","content":system},{"role":"user","content":user}],
    temperature=0.2
)
raw_json = resp.choices[0].message.content.strip()

# ---- 4) Validate with Pydantic ----
summary_obj = validate_portfolio_summary(raw_json)

# Optional: craft a short human-readable blurb
narrative = dedent(f"""
# Portfolio Summary
- **DSO (90d):** {summary_obj.dso_days_90d or "N/A"} days  
- **Aging (30+/60+/90+):** {summary_obj.aging_overview}  
- **High Utilization (>=90%)**: {len(summary_obj.high_utilization_accounts)} accounts  
- **Failed Promises (30d):** {summary_obj.failed_promises_30d}

**Risks:** {", ".join([r.name for r in summary_obj.key_risks]) or "None flagged"}  
**Notes:** {summary_obj.notes or "—"}
""").strip()

# ---- 5) Persist to MySQL ----
cur.execute(
    """
    INSERT INTO summaries(scope, summary_json, narrative_markdown, validation_ok, kpi_snapshot_json, confidence_score)
    VALUES ('portfolio', %s, %s, TRUE, %s, %s)
    """,
    (summary_obj.model_dump_json(), narrative, json.dumps(kpi_snapshot), 0.90)
)
conn.commit()

print("Saved summary_id:", cur.lastrowid)
cur.close(); conn.close()
