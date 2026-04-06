# Agentic Multi-Source Retrieval System

A CLI tool that answers business questions by routing queries to the right
data sources, computing insights with pandas, and producing cited answers —
**all without relying on an LLM for reasoning**.

---

## Quick Start

### 1. Install the only required dependency

```bash
pip install pandas
```

*(Optional — only needed for natural-language answer formatting)*
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

### 2. Run

```bash
python agent.py
```

### 3. Ask questions

```
Enter your question: Why did revenue decrease last month?
Enter your question: Compare salary expenses with sales
Enter your question: Provide a brief summary of business performance
Enter your question: Which region has the highest returns?
Enter your question: How has headcount changed over the past few months?
```

Type `exit` or `quit` to stop.

---

## Example Output

```
Enter your question: Why did revenue decrease last month?

[Agent Decision]
  - Using sales.csv for numerical revenue and sales analysis
  - Using report.txt for narrative context and explanations

──────────────────────────────────────────────────
--- ANSWER ---

1. Total net revenue was $136,750 in 2024-08 and $112,160 in 2024-11
   (decreased by 18.0%).
2. In 2024-11, the top-performing region was North ($76,460 net revenue)
   and the weakest was South ($21,350).
3. South region net revenue fell from $41,800 (2024-08) to $21,350
   (2024-11), a 48.9% decline.
4. The South region has been the primary driver of revenue deterioration...

--- SOURCES ---
  Source: sales.csv (rows 2–25)
  Source: sales.csv (rows 20–25)
  Source: report.txt (paragraph 1)
  Source: report.txt (paragraph 3)
──────────────────────────────────────────────────
```

---

## Project Structure

```
project/
├── agent.py          # Main CLI application
├── README.md         # This file
└── data/
    ├── sales.csv     # Monthly sales by region and product
    ├── payroll.csv   # Monthly payroll by department
    └── report.txt    # Narrative business performance report
```

---

## Approach

The system follows a strict three-stage pipeline. **No LLM is involved in
stages 1 or 2.**

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: decide_sources()      │  Keyword routing → pick CSV/TXT files
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2: retrieve_*()          │  pandas + text parsing → raw insights
│  (pure Python, no LLM)          │  with citations attached
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 3: format answer         │  Plain text (always) or
│  LLM = formatter only           │  LLM reformats pre-built insights
└─────────────────────────────────┘
```

---

## Source Selection Logic (`decide_sources`)

Routing is keyword-based. The function scans the query for known terms
and maps them to source files:

| Keywords in query                                          | Source(s) used              |
|------------------------------------------------------------|-----------------------------|
| `revenue`, `sales`, `units`, `product`, `region`, `widget`| `sales.csv`                 |
| `salary`, `payroll`, `expense`, `employee`, `headcount`   | `payroll.csv`               |
| `why`, `summary`, `explain`, `overview`, `trend`          | `report.txt`                |
| `compare`                                                  | `sales.csv` + `payroll.csv` |
| Any reasoning word (`why`, `explain`, `cause`, …)         | always adds `report.txt`    |
| No match                                                   | all three sources           |

The decision is printed before retrieval:

```
[Agent Decision]
  - Using sales.csv for numerical revenue and sales analysis
  - Using report.txt for narrative context and explanations
```

---

## Retrieval Logic

### CSV files (`retrieve_sales`, `retrieve_payroll`)

All computation uses **pandas** directly — no LLM:

- Monthly totals and percentage change (`groupby` + `sum`)
- Region/department breakdowns for the latest month
- Product-level comparisons
- Headcount growth over time
- Revenue vs payroll ratio per month

Each insight records the exact DataFrame rows it was derived from,
producing citations like `sales.csv (rows 14–19)`.

### Text file (`retrieve_report`)

- File is split on blank lines into numbered paragraphs
- Header lines (all-caps) are filtered out
- Each paragraph is scored against query keywords
- Paragraphs with at least one matching word are returned
- Paragraph 1 (executive summary) is always included for
  reasoning-type queries (`why`, `explain`, `summary`)

Citations reference the paragraph number: `report.txt (paragraph 3)`.

---

## Citation Method

Citations are generated **during retrieval**, not by an LLM.

Every insight dictionary has two keys:

```python
{
    "text":     "South region net revenue fell from $41,800 to $21,350 (48.9% decline).",
    "citation": "sales.csv (rows 8–13)",
}
```

`_row_range()` maps pandas index positions back to 1-based CSV row numbers
(header = row 1, first data row = row 2).  For text paragraphs, the
paragraph counter assigned during splitting is used directly.

The `--- SOURCES ---` block deduplicates citations before printing.

---

## LLM Usage Policy

> **Reasoning is done entirely without an LLM.**

The LLM (Claude) is only invoked when `ANTHROPIC_API_KEY` is set, and only
to reformat the already-computed bullet-point insights into a readable
paragraph.  The system prompt explicitly forbids the model from adding new
data or performing calculations.

If the API key is absent, or if the API call fails for any reason, the
system falls back to a numbered plain-text list.  The citations and
source-selection logic are identical in both paths.

---

## Data Description

| File           | Contents                                                            |
|----------------|---------------------------------------------------------------------|
| `sales.csv`    | Monthly sales Aug–Nov 2024 by region, product, units, and revenue  |
| `payroll.csv`  | Monthly payroll Aug–Nov 2024 by department, headcount, and expense |
| `report.txt`   | Narrative Q4 2024 report: summary, regional analysis, outlook      |

---

## Requirements

- Python 3.10+
- `pandas` (`pip install pandas`)
- *(optional)* `anthropic` + `ANTHROPIC_API_KEY` for natural-language formatting
