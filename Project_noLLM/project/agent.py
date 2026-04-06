"""
Agentic Multi-Source Retrieval System
======================================
All reasoning, calculations, and retrieval are performed in pure Python
using pandas. An LLM (Claude) is optionally used ONLY to reformat the
pre-computed insights into natural language. The system works fully
without any API key.

Usage:
    python agent.py
"""

import os
import re
import textwrap
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
SALES_CSV   = DATA_DIR / "sales.csv"
PAYROLL_CSV = DATA_DIR / "payroll.csv"
REPORT_TXT  = DATA_DIR / "report.txt"

CURRENCY = "₹"


# ══════════════════════════════════════════════════════════════════════════
# 1.  AGENT DECISION LOGIC
# ══════════════════════════════════════════════════════════════════════════

# Keyword → source mapping (order matters: more specific first)
SOURCE_RULES = [
    ({"compare"},                    ["sales.csv", "payroll.csv"]),
    ({"revenue", "sales", "units",
      "product", "region", "widget",
      "returns", "income"},          ["sales.csv"]),
    ({"salary", "payroll", "wage",
      "headcount", "employee",
      "bonus", "department",
      "staff", "labour", "labor",
      "expense"},                    ["payroll.csv"]),
    ({"why", "summary", "explain",
      "overview", "reason", "cause",
      "trend", "performance",
      "report", "highlight"},        ["report.txt"]),
]

# Reasoning triggers that always add report.txt for context
REASONING_TRIGGERS = {"why", "explain", "reason", "cause", "summary", "overview"}


def decide_sources(query: str) -> list[str]:
    """
    Keyword-based agentic routing.  Returns an ordered list of source
    file names to use for this query.
    """
    q_words = set(re.findall(r"[a-z]+", query.lower()))
    selected: list[str] = []

    for keywords, sources in SOURCE_RULES:
        if keywords & q_words:                  # any keyword matches
            for s in sources:
                if s not in selected:
                    selected.append(s)

    # Always pull in report.txt when the query asks for reasoning
    if REASONING_TRIGGERS & q_words and "report.txt" not in selected:
        selected.append("report.txt")

    # Fallback: use all sources if nothing matched
    if not selected:
        selected = ["sales.csv", "payroll.csv", "report.txt"]

    return selected


# ══════════════════════════════════════════════════════════════════════════
# 2.  STRUCTURED DATA RETRIEVAL  (pandas)
# ══════════════════════════════════════════════════════════════════════════

def _row_range(df: pd.DataFrame, mask: pd.Series) -> str:
    """Return a human-readable row range string for the matched rows."""
    indices = df.index[mask].tolist()
    if not indices:
        return "no rows"
    # +2 because CSV row 1 is header, pandas index starts at 0
    rows = [i + 2 for i in indices]
    return f"rows {rows[0]}–{rows[-1]}"


def retrieve_sales(query: str) -> list[dict]:
    """
    Load sales.csv, run pandas logic, return a list of insight dicts:
        { "text": str, "citation": str }
    """
    q = query.lower()
    df = pd.read_csv(SALES_CSV)
    insights: list[dict] = []

    # ── Monthly net revenue totals & trend ────────────────────────────────
    monthly = (
        df.groupby("month")["net_revenue"]
        .sum()
        .sort_index()
    )
    months = monthly.index.tolist()
    if len(months) >= 2:
        first_month, last_month   = months[0],  months[-1]
        first_rev,   last_rev     = monthly.iloc[0], monthly.iloc[-1]
        pct_change = ((last_rev - first_rev) / first_rev) * 100

        mask_all = pd.Series([True] * len(df), index=df.index)
        insights.append({
            "text": (
                f"Total net revenue was {CURRENCY}{first_rev:,.0f} in {first_month} "
                f"and {CURRENCY}{last_rev:,.0f} in {last_month} "
                f"({'decreased' if pct_change < 0 else 'increased'} "
                f"by {abs(pct_change):.1f}%)."
            ),
            "citation": f"sales.csv ({_row_range(df, mask_all)})",
        })

    # ── Region breakdown for the most recent month ───────────────────────
    latest_month = df["month"].max()
    mask_latest  = df["month"] == latest_month
    region_rev   = (
        df[mask_latest]
        .groupby("region")["net_revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    top_region    = region_rev.idxmax()
    bottom_region = region_rev.idxmin()
    insights.append({
        "text": (
            f"In {latest_month}, the top-performing region was {top_region} "
            f"({CURRENCY}{region_rev[top_region]:,.0f} net revenue) and the weakest "
            f"was {bottom_region} ({CURRENCY}{region_rev[bottom_region]:,.0f})."
        ),
        "citation": f"sales.csv ({_row_range(df, mask_latest)})",
    })

    # ── Returns analysis ──────────────────────────────────────────────────
    if any(w in q for w in ("return", "returns")):
        returns_by_region = df.groupby("region")["returns"].sum().sort_values(ascending=False)
        top_returns_region = returns_by_region.idxmax()
        mask_returns = df["region"] == top_returns_region
        insights.append({
            "text": (
                f"Highest cumulative returns came from the {top_returns_region} region "
                f"({CURRENCY}{returns_by_region[top_returns_region]:,.0f} total returns)."
            ),
            "citation": f"sales.csv ({_row_range(df, mask_returns)})",
        })

    # ── Product comparison ────────────────────────────────────────────────
    if any(w in q for w in ("product", "widget", "compare", "comparison")):
        product_rev = df.groupby("product")["net_revenue"].sum().sort_values(ascending=False)
        lines = [f"{prod}: {CURRENCY}{rev:,.0f}" for prod, rev in product_rev.items()]
        mask_all = pd.Series([True] * len(df), index=df.index)
        insights.append({
            "text": "Net revenue by product (all months): " + " | ".join(lines) + ".",
            "citation": f"sales.csv ({_row_range(df, mask_all)})",
        })

    # ── South region deep-dive (common for 'why' queries) ────────────────
    if any(w in q for w in ("why", "decrease", "decline", "drop", "fall")):
        mask_south = df["region"] == "South"
        south_monthly = (
            df[mask_south]
            .groupby("month")["net_revenue"]
            .sum()
            .sort_index()
        )
        if len(south_monthly) >= 2:
            s_first = south_monthly.iloc[0]
            s_last  = south_monthly.iloc[-1]
            s_pct   = ((s_last - s_first) / s_first) * 100
            insights.append({
                "text": (
                    f"South region net revenue fell from {CURRENCY}{s_first:,.0f} "
                    f"({south_monthly.index[0]}) to {CURRENCY}{s_last:,.0f} "
                    f"({south_monthly.index[-1]}), a {abs(s_pct):.1f}% decline."
                ),
                "citation": f"sales.csv ({_row_range(df, mask_south)})",
            })

    return insights


def retrieve_payroll(query: str) -> list[dict]:
    """
    Load payroll.csv, run pandas logic, return insight dicts.
    """
    q = query.lower()
    df = pd.read_csv(PAYROLL_CSV)
    insights: list[dict] = []

    # ── Total payroll trend ───────────────────────────────────────────────
    monthly = (
        df.groupby("month")["total_expense"]
        .sum()
        .sort_index()
    )
    if len(monthly) >= 2:
        first_exp = monthly.iloc[0]
        last_exp  = monthly.iloc[-1]
        pct = ((last_exp - first_exp) / first_exp) * 100
        mask_all = pd.Series([True] * len(df), index=df.index)
        insights.append({
            "text": (
                f"Total payroll expense rose from {CURRENCY}{first_exp:,.0f} "
                f"({monthly.index[0]}) to {CURRENCY}{last_exp:,.0f} "
                f"({monthly.index[-1]}), an increase of {pct:.1f}%."
            ),
            "citation": f"payroll.csv ({_row_range(df, mask_all)})",
        })

    # ── Department breakdown for latest month ────────────────────────────
    latest_month = df["month"].max()
    mask_latest  = df["month"] == latest_month
    dept_exp = (
        df[mask_latest]
        .groupby("department")["total_expense"]
        .sum()
        .sort_values(ascending=False)
    )
    lines = [f"{dept}: {CURRENCY}{exp:,.0f}" for dept, exp in dept_exp.items()]
    insights.append({
        "text": f"Payroll breakdown in {latest_month}: " + " | ".join(lines) + ".",
        "citation": f"payroll.csv ({_row_range(df, mask_latest)})",
    })

    # ── Headcount growth ─────────────────────────────────────────────────
    if any(w in q for w in ("headcount", "employee", "staff", "hire", "hiring")):
        hc = df.groupby("month")["employee_count"].sum().sort_index()
        if len(hc) >= 2:
            mask_all = pd.Series([True] * len(df), index=df.index)
            insights.append({
                "text": (
                    f"Total headcount grew from {hc.iloc[0]} employees "
                    f"({hc.index[0]}) to {hc.iloc[-1]} ({hc.index[-1]})."
                ),
                "citation": f"payroll.csv ({_row_range(df, mask_all)})",
            })

    # ── Salary vs sales comparison ────────────────────────────────────────
    if any(w in q for w in ("compare", "vs", "versus", "against", "sales")):
        sales_df  = pd.read_csv(SALES_CSV)
        sales_monthly = sales_df.groupby("month")["net_revenue"].sum().sort_index()
        payroll_monthly = df.groupby("month")["total_expense"].sum().sort_index()

        common_months = sales_monthly.index.intersection(payroll_monthly.index)
        rows = []
        for m in common_months:
            rev  = sales_monthly[m]
            exp  = payroll_monthly[m]
            ratio = (exp / rev) * 100
            rows.append(f"{m}: revenue {CURRENCY}{rev:,.0f} vs payroll {CURRENCY}{exp:,.0f} ({ratio:.1f}% ratio)")

        mask_all_p = pd.Series([True] * len(df), index=df.index)
        mask_all_s = pd.Series([True] * len(sales_df), index=sales_df.index)
        insights.append({
            "text": "Revenue vs payroll comparison by month — " + " | ".join(rows) + ".",
            "citation": (
                f"sales.csv ({_row_range(sales_df, mask_all_s)}), "
                f"payroll.csv ({_row_range(df, mask_all_p)})"
            ),
        })

    return insights


# ══════════════════════════════════════════════════════════════════════════
# 3.  UNSTRUCTURED DATA RETRIEVAL  (text / paragraph matching)
# ══════════════════════════════════════════════════════════════════════════

def retrieve_report(query: str) -> list[dict]:
    """
    Load report.txt, split into paragraphs, return paragraphs that
    contain at least one query keyword.  Always includes the first
    paragraph (executive summary) when a summary/why query is detected.
    """
    text = REPORT_TXT.read_text(encoding="utf-8")

    # Split on blank lines; strip whitespace; drop header lines
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Skip the document header (lines that are all-caps titles / dates)
    paras: list[str] = []
    for p in raw_paras:
        first_line = p.split("\n")[0]
        if re.match(r"^[A-Z0-9 \-–—/:,]+$", first_line) and len(first_line) < 80:
            continue        # header line — skip
        paras.append(p)

    q_words = set(re.findall(r"[a-z]+", query.lower()))

    insights: list[dict] = []
    for i, para in enumerate(paras, start=1):
        para_words = set(re.findall(r"[a-z]+", para.lower()))

        # Always include para 1 (exec summary) for why/summary queries
        include = (i == 1 and REASONING_TRIGGERS & q_words) or bool(q_words & para_words)

        if include:
            # Truncate very long paragraphs for readability
            display = para if len(para) <= 400 else para[:397] + "…"
            insights.append({
                "text": display,
                "citation": f"report.txt (paragraph {i})",
            })

    return insights


def clean_citations(citations: list[str]) -> list[str]:
    """Merge overlapping row ranges and remove duplicates."""
    import re
    file_ranges = {}
    other_cites = set()
    
    # Flatten any comma-separated citations
    flat_citations = []
    for c in citations:
        for part in c.split(","):
            if part.strip():
                flat_citations.append(part.strip())
                
    for cite in flat_citations:
        match = re.match(r"(.*?)\s*\(rows\s+(\d+)[-–—](\d+)\)", cite)
        if match:
            file, start, end = match.groups()
            start, end = int(start), int(end)
            if file not in file_ranges:
                file_ranges[file] = []
            file_ranges[file].append((start, end))
        else:
            other_cites.add(cite)
            
    final_cites = []
    for file, ranges in file_ranges.items():
        ranges.sort()
        merged = []
        for r in ranges:
            if not merged:
                merged.append(r)
            else:
                last = merged[-1]
                if r[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], r[1]))
                else:
                    merged.append(r)
                    
        for m in merged:
            final_cites.append(f"{file} (rows {m[0]}–{m[1]})")
            
    for c in other_cites:
        final_cites.append(c)
        
    return sorted(final_cites)


def compute_comparison() -> list[dict]:
    """Dedicated function to strictly compare total revenue and total payroll."""
    import pandas as pd
    sales_df = pd.read_csv(SALES_CSV)
    payroll_df = pd.read_csv(PAYROLL_CSV)
    
    total_rev = sales_df["net_revenue"].sum()
    total_pay = payroll_df["total_expense"].sum()
    
    mask_s = pd.Series([True]*len(sales_df), index=sales_df.index)
    mask_p = pd.Series([True]*len(payroll_df), index=payroll_df.index)
    
    return [{
        "text": f"Total Revenue was {CURRENCY}{total_rev:,.0f} vs Total Payroll of {CURRENCY}{total_pay:,.0f}.",
        "citation": f"sales.csv ({_row_range(sales_df, mask_s)}), payroll.csv ({_row_range(payroll_df, mask_p)})"
    }]


def extract_cause_phrase(text: str) -> str:
    """Extracts the specific cause phrase from a paragraph."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    
    # Priority keywords that usually precede the actual cause
    cause_kws = ["driven by", "due to", "because of", "result of", "led to", "include ", "caused by"]
    
    for s in sentences:
        s_lower = s.lower()
        for kw in cause_kws:
            idx = s_lower.find(kw)
            if idx != -1:
                return s[idx + len(kw):].strip().rstrip('.')
                
    return sentences[0].rstrip('.')


def get_root_cause_paragraph(paragraphs: list[dict]) -> dict:
    """Takes a list of insights (report paragraphs) and finds the best root cause."""
    keywords = ["cause", "reason", "due to", "because", "driven by", "result of", "led to"]
    
    report_items = [i for i in paragraphs if "report.txt" in i["citation"]]
    if not report_items:
        return None
        
    # 1. Look for explicit cause keywords
    for item in report_items:
        text_lower = item["text"].lower()
        if any(kw in text_lower for kw in keywords):
            return item
            
    # 2. Fallback to paragraph 2 or 3 (avoid 1 which is summary)
    for item in report_items:
        if "paragraph 2" in item["citation"] or "paragraph 3" in item["citation"]:
            return item
            
    return report_items[-1]

def generate_business_insight(query: str, data: dict) -> str:
    """Analyze query intent and return meaningful business-level conclusions."""
    q_lower = query.lower()
    
    if any(w in q_lower for w in ["anomaly", "unusual"]):
        if data.get("payroll_gt_revenue"):
            return "An anomaly is observed where payroll exceeds revenue across all months."
        return "An anomaly is observed with a sharp regional performance drop."
        
    elif any(w in q_lower for w in ["recommend", "should", "action"]):
        if data.get("rev_decreasing") and data.get("pay_increasing"):
            return "The company should reduce operational costs and improve revenue streams."
        return "The company should maintain operational strategies."
            
    elif any(w in q_lower for w in ["risk", "concern", "sustainable", "health"]):
        if data.get("payroll_sum", 0) > data.get("revenue_sum", 0):
            return "The business is financially unsustainable due to higher costs than revenue."
        return "The business exhibits signs of financial pressure requiring review."
            
    elif any(w in q_lower for w in ["region", "north", "south", "east", "west"]):
        best = data.get("best_region", "North")
        worst = data.get("worst_region", "South")
        return f"{best} performs strongest while {worst} shows decline, requiring attention."
        
    elif any(w in q_lower for w in ["why", "reason", "cause", "explain"]):
        cause = data.get("cause", "")
        if "competiti" in cause.lower() or "supply" in cause.lower() or "disrupt" in cause.lower():
            return "indicating market challenges."
        return "indicating operational shifts."

    elif any(w in q_lower for w in ["compare", "vs", "comparison", "against", "higher", "lower", "department"]):
        if data.get("payroll_sum", 0) > data.get("revenue_sum", 0):
            return "Payroll expenses exceed revenue, indicating high operational costs."
        return "Revenue sustains payroll, indicating stable profit margins."
        
    elif any(w in q_lower for w in ["trend", "change", "increase", "decrease", "growth", "evolved"]):
        if data.get("is_rev", True):
            if data.get("rev_decreasing"):
                return "indicating declining performance."
            return "indicating positive operational growth."
        else:
            if data.get("pay_increasing"):
                return "indicating rising operational burden."
            return "indicating reducing overhead."
            
    elif any(w in q_lower for w in ["summary", "overview"]):
        return "Financial sustainability requires balancing these operational gaps."
        
    return "Highlights key metrics for executive review."

def combine_insight(factual_text: str, insight: str) -> str:
    factual_text = factual_text.strip().rstrip(".")
    if insight.startswith("indicating") or insight.startswith("requiring"):
        return f"{factual_text}, {insight}"
    else:
        return f"{factual_text}. {insight}"


def generate_answer(query: str, insights: list[dict]) -> tuple[str, list[dict]]:
    """
    Filter insights based on relevance and format into a clean, concise answer.
    """
    q_lower = query.lower()
    import pandas as pd
    sales_df = pd.read_csv(SALES_CSV)
    payroll_df = pd.read_csv(PAYROLL_CSV)
    
    rev_m = sales_df.groupby("month")["net_revenue"].sum()
    pay_m = payroll_df.groupby("month")["total_expense"].sum()
    rev_sum = rev_m.sum()
    pay_sum = pay_m.sum()
    
    data = {
        "revenue_sum": rev_sum,
        "payroll_sum": pay_sum,
        "rev_decreasing": rev_m.iloc[-1] < rev_m.iloc[0] if len(rev_m) > 1 else False,
        "pay_increasing": pay_m.iloc[-1] > pay_m.iloc[0] if len(pay_m) > 1 else False,
        "payroll_gt_revenue": all(pay_m[m] > rev_m[m] for m in rev_m.index if m in pay_m) if not pay_m.empty else False,
        "cause": "",
        "is_rev": True
    }
    
    try:
        reg_sum = sales_df.groupby("region")["net_revenue"].sum().sort_values()
        data["best_region"] = reg_sum.index[-1]
        data["worst_region"] = reg_sum.index[0]
    except:
        data["best_region"] = "North"
        data["worst_region"] = "South"

    intent = "general"
    if any(w in q_lower for w in ["anomaly", "unusual"]):
        intent = "anomaly"
    elif any(w in q_lower for w in ["recommend", "action", "should", "strategic"]):
        intent = "decision"
    elif any(w in q_lower for w in ["why", "reason", "cause", "explain"]):
        intent = "reasoning"
    elif any(w in q_lower for w in ["department", "payroll breakdown", "departments"]):
        intent = "department_comparison"
    elif any(w in q_lower for w in ["region", "north", "south", "east", "west"]):
        intent = "region_comparison"
    elif any(w in q_lower for w in ["higher", "cost vs revenue", "salary vs revenue", "compare salary and revenue", "highest", "lowest", "gap"]):
        intent = "cost_vs_revenue"
    elif any(w in q_lower for w in ["cost", "expenses", "operational cost"]):
        intent = "operational_cost"
    elif any(w in q_lower for w in ["risk", "concern", "sustainable", "health"]):
        intent = "financial_health"
    elif any(w in q_lower for w in ["summary", "overview"]):
        intent = "summary"
    elif any(w in q_lower for w in ["compare", "vs", "comparison", "against"]):
        intent = "comparison"
    elif any(w in q_lower for w in ["trend", "change", "increase", "decrease", "growth", "evolved"]):
        intent = "trend"


    if intent == "anomaly":
        anomalies = []
        for m in rev_m.index:
            if m in pay_m.index and pay_m[m] > rev_m[m]:
                anomalies.append(f"In {m}, payroll expense ({CURRENCY}{pay_m[m]:,.0f}) exceeded revenue ({CURRENCY}{rev_m[m]:,.0f}).")
                
        # check sharp region drop
        s_df = sales_df[sales_df["region"] == "South"].groupby("month")["net_revenue"].sum().sort_index()
        if len(s_df) > 1:
            pct = abs((s_df.iloc[-1] - s_df.iloc[0]) / s_df.iloc[0] * 100)
            if pct > 30:
                anomalies.append(f"South region revenue sharply dropped by {pct:.1f}%.")
                
        if anomalies:
            fact = " ".join(anomalies)
            insight = generate_business_insight(query, data)
            text = f"Anomalies detected: {fact} {insight}"
            mask_s = pd.Series([True] * len(sales_df), index=sales_df.index)
            mask_p = pd.Series([True] * len(payroll_df), index=payroll_df.index)
            cites = f"sales.csv ({_row_range(sales_df, mask_s)}), payroll.csv ({_row_range(payroll_df, mask_p)})"
            item = {"text": text, "citation": cites}
            return f"- {text}", [item]

    elif intent == "decision":
        prob = [i for i in insights if "total net revenue was" in i["text"].lower() and "decreased by" in i["text"].lower()]
        rec = [i for i in insights if "report.txt" in i["citation"] and any(kw in i["text"].lower() for kw in ["restructur", "reduc", "recapture", "immediate attention"])]
        
        combined = []
        if prob: 
            fact_str = prob[0]["text"]
            insight = generate_business_insight(query, data)
            ans = combine_insight(fact_str, insight)
            prob[0]["text"] = ans
            combined.append(prob[0])
            
        if rec:
            rec_text = rec[-1]["text"]
            sents = [s.strip() + "." for s in rec_text.split(". ") if s.strip()]
            rec_sent = next((s for s in sents if any(kw in s.lower() for kw in ["restructur", "reduc", "recapture", "immediate attention"])), sents[-1])
            rec_item = {"text": f"Recommendation: {rec_sent}", "citation": rec[-1]["citation"]}
            combined.append(rec_item)
            
        if combined:
            lines = [f"- {i['text']}" for i in combined[:2]]
            return "\n".join(lines), combined[:2]

    elif intent == "financial_health":
        rev_trend = [i for i in insights if "total net revenue was" in i["text"].lower() and "decreased by" in i["text"].lower()]
        
        combined = []
        if rev_trend:
            insight = generate_business_insight(query, data)
            ans = combine_insight(rev_trend[0]["text"], insight)
            rev_trend[0]["text"] = ans
            combined.append(rev_trend[0])
        
        if combined:
            lines = [f"- {i['text'].strip()}" for i in combined[:2]]
            return "\n".join(lines), combined[:2]

    elif intent == "operational_cost":
        comp = [i for i in insights if "total payroll expense" in i["text"].lower() or "payroll expense" in i["text"].lower()]
        if comp:
            data["is_rev"] = False
            insight = generate_business_insight("trend", data)
            ans = combine_insight(comp[0]["text"], insight)
            comp[0]["text"] = ans
            return f"- {ans}", comp[:1]
            
    elif intent == "summary":
        rev_trend = [i for i in insights if "total net revenue was" in i["text"].lower() and "decreased by" in i["text"].lower()]
        cost_trend = [i for i in insights if "total payroll expense" in i["text"].lower()]
        reg_perf = [i for i in insights if "top-performing region" in i["text"].lower()]
        
        combined = []
        if rev_trend: combined.append(rev_trend[0])
        if cost_trend: combined.append(cost_trend[0])
        if reg_perf: combined.append(reg_perf[0])
        
        if combined:
            fact = f"Revenue trended {'down' if data['rev_decreasing'] else 'up'} and payroll {'rose' if data['pay_increasing'] else 'fell'}, with {data['best_region']} as the top performer."
            insight = generate_business_insight(query, data)
            ans = f"- {fact}\n- Insight: {insight}"
            return ans, combined[:2]

    elif intent == "department_comparison":
        comp = [i for i in insights if "payroll breakdown" in i["text"].lower()]
        if not comp:
            comp = [i for i in insights if "engineering:" in i["text"].lower()]
        if comp:
            insight = generate_business_insight(query, data)
            ans = combine_insight(comp[0]["text"], insight)
            comp[0]["text"] = ans
            return f"- {ans}", comp[:1]

    elif intent == "region_comparison":
        comp = [i for i in insights if "top-performing region" in i["text"].lower() or "weakest" in i["text"].lower()]
        if not comp:
             comp = [i for i in insights if "region" in i["text"].lower() and "sales.csv" in i["citation"].lower()]
        if comp:
            insight = generate_business_insight(query, data)
            ans = combine_insight(comp[0]["text"], insight)
            comp[0]["text"] = ans
            return f"- {ans}", comp[:1]

    elif intent == "cost_vs_revenue" or intent == "comparison":
        comp_insights = list(compute_comparison())
        if any(w in q_lower for w in ["higher", "lower", "cost vs revenue"]):
            ans_fact = "Revenue is higher than payroll" if rev_sum > pay_sum else "Payroll is higher than revenue"
            comp_insights[0]["text"] = f"{ans_fact} ({comp_insights[0]['text']})"
            
        insight = generate_business_insight(query, data)
        ans = combine_insight(comp_insights[0]["text"], insight)
        comp_insights[0]["text"] = ans
        return f"- {ans}", comp_insights

    elif intent == "reasoning":
        csv_items = [i for i in insights if ".csv" in i["citation"] and any(kw in i["text"].lower() for kw in ["fell", "decreased", "decline", "rose", "increase", "grew", "was"])]
        metric_item = csv_items[0] if csv_items else None
        
        txt_item = get_root_cause_paragraph(insights)
        
        if metric_item and txt_item:
            cause_phrase = extract_cause_phrase(txt_item["text"])
            data["cause"] = cause_phrase
            insight = generate_business_insight(query, data)
            
            factual = f"{metric_item['text'].rstrip('.')} due to {cause_phrase}"
            ans = combine_insight(factual, insight)
            metric_item["text"] = ans
            return f"- {ans}", [metric_item, txt_item]
        elif txt_item:
            return f"- {txt_item['text']}", [txt_item]

    # Category matching for other intents/fallbacks
    categories = {
        "headcount": ["headcount", "employee", "staff", "hire", "hiring"],
        "payroll": ["payroll", "salary", "expense", "wage", "bonus"],
        "revenue": ["revenue", "sales", "income", "sell", "sold"],
        "return": ["return", "returns"],
        "product": ["product", "widget"]
    }
    
    active_categories = set(cat for cat, words in categories.items() if any(w in q_lower for w in words))
    
    filtered = []
    for item in insights:
        text_lower = item["text"].lower()
        if active_categories:
            if any(any(w in text_lower for w in categories[cat]) for cat in active_categories):
                filtered.append(item)
        else:
            filtered.append(item)

    if not filtered:
        filtered = insights

    if intent == "trend":
        trend_items = [i for i in filtered if "decreased by" in i["text"].lower() or "increase of" in i["text"].lower() or "fell from" in i["text"].lower() or "grew from" in i["text"].lower()]
        if trend_items:
            data["is_rev"] = "revenue" in trend_items[0]["text"].lower()
            insight = generate_business_insight(query, data)
            ans = combine_insight(trend_items[0]["text"], insight)
            trend_items[0]["text"] = ans
            filtered = trend_items[:1]

    final_insights = filtered[:2]
    answer_lines = [f"- {item['text'].strip()}" for item in final_insights]
    return "\n".join(answer_lines), final_insights


# ══════════════════════════════════════════════════════════════════════════
# 5.  MAIN AGENT
# ══════════════════════════════════════════════════════════════════════════

def run_query(query: str) -> None:
    """Full pipeline for a single user query."""
    sources = decide_sources(query)

    print()
    print("[Agent Decision]")
    for src in sources:
        print(f"- Using {src}")

    all_insights: list[dict] = []

    for src in sources:
        if src == "sales.csv":
            all_insights.extend(retrieve_sales(query))
        elif src == "payroll.csv":
            all_insights.extend(retrieve_payroll(query))
        elif src == "report.txt":
            all_insights.extend(retrieve_report(query))

    if not all_insights:
        print("\n[No relevant data found for this query.]\n")
        return

    # Pass through our smart answer generator
    answer_text, filtered_insights = generate_answer(query, all_insights)

    print()
    print("--- ANSWER ---")
    for line in answer_text.splitlines():
        if line.startswith("- "):
            wrapped = textwrap.fill(line, width=70, subsequent_indent="  ")
            print(wrapped)
        else:
            print(textwrap.fill(line, width=70))

    print()
    print("--- SOURCES ---")
    raw_citations = [item["citation"] for item in filtered_insights]
    cleaned = clean_citations(raw_citations)
    for c in cleaned:
        print(f"Source: {c}")
    print()


def main() -> None:
    print("Agentic Multi-Source Retrieval System")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        print()
        try:
            query = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        run_query(query)


if __name__ == "__main__":
    main()
