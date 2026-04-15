import json
import os
import sys
from datetime import datetime
import math

RESULTS_FILE = "ragas_results.json"
REPORT_FILE  = "ragas_report.html"


# ─────────────────────────────
# Utility: normalize values
# ─────────────────────────────
def safe_val(val):
    if val is None:
        return 0.0
    if isinstance(val, float) and math.isnan(val):
        return 0.0
    return val


def is_nan(val):
    return isinstance(val, float) and math.isnan(val)


# ─────────────────────────────
# Score helpers
# ─────────────────────────────
def score_color(val: float) -> str:
    val = safe_val(val)

    if val >= 0.75:
        return "#1D9E75"
    if val >= 0.50:
        return "#BA7517"
    return "#E24B4A"


def score_label(val: float) -> str:
    if val is None or is_nan(val):
        return "N/A"

    if val >= 0.75:
        return "Good"
    if val >= 0.50:
        return "Fair"
    return "Poor"


def bar_html(val: float, color: str) -> str:
    val = safe_val(val)
    pct = int(round(val * 100))

    return (
        f'<div style="background:#eee;border-radius:6px;height:10px;width:100%;margin-top:4px">'
        f'<div style="background:{color};width:{pct}%;height:10px;border-radius:6px;'
        f'transition:width 0.4s"></div></div>'
    )


# ─────────────────────────────
# Main report generator
# ─────────────────────────────
def generate_report(data: dict) -> str:
    agg  = data["aggregate_scores"]
    rows = data["per_question"]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")

    nan_counter = 0

    metric_meta = [
        ("faithfulness",      "Faithfulness",
         "Is the answer grounded in the retrieved context? Penalises hallucination."),
        ("answer_relevancy",  "Answer Relevancy",
         "Does the answer actually address the question asked?"),
        ("context_precision", "Context Precision",
         "Are the retrieved chunks relevant to the question?"),
        ("context_recall",    "Context Recall",
         "Were all relevant document passages retrieved?"),
    ]

    # ── Aggregate cards ──
    cards_html = ""
    for key, label, desc in metric_meta:
        val_raw = agg.get(key)

        if is_nan(val_raw):
            nan_counter += 1

        val   = safe_val(val_raw)
        col   = score_color(val_raw)
        lbl   = score_label(val_raw)
        disp  = f"{val:.4f}" if val_raw is not None and not is_nan(val_raw) else "N/A"

        cards_html += f"""
        <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;
                    padding:20px 24px;flex:1;min-width:180px">
          <div style="font-size:13px;color:#666;margin-bottom:6px">{label}</div>
          <div style="font-size:28px;font-weight:600;color:{col}">{disp}</div>
          <div style="font-size:12px;color:{col};margin-bottom:8px">{lbl}</div>
          {bar_html(val_raw, col)}
          <div style="font-size:11px;color:#999;margin-top:10px;line-height:1.4">{desc}</div>
        </div>"""

    # ── Per-question table rows ──
    table_rows = ""
    for item in rows:
        sc   = item["scores"]
        cols = ""

        for key, _, _ in metric_meta:
            v_raw = sc.get(key)

            if is_nan(v_raw):
                nan_counter += 1

            v   = safe_val(v_raw)
            col = score_color(v_raw)
            txt = f"{v:.3f}" if v_raw is not None and not is_nan(v_raw) else "—"

            cols += (f'<td style="text-align:center;font-weight:600;color:{col};'
                     f'padding:10px 8px">{txt}</td>')

        q  = item["question"]
        a  = item["answer"][:160] + ("…" if len(item["answer"]) > 160 else "")
        gt_raw = item.get("ground_truth", "")
        gt = gt_raw[:120] + ("…" if len(gt_raw) > 120 else "")

        table_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="padding:10px 8px;color:#555;font-size:12px;min-width:30px;
                     text-align:center">{item['id']}</td>
          <td style="padding:10px 8px;font-size:13px;max-width:280px">{q}</td>
          <td style="padding:10px 8px;font-size:12px;color:#444;max-width:220px">{a}</td>
          {cols}
        </tr>"""

    # ── Interpretation ──
    interp_html = ""
    for key, label, desc in metric_meta:
        val_raw = agg.get(key)

        if val_raw is None or is_nan(val_raw):
            continue

        val = val_raw
        col = score_color(val)

        if val >= 0.75:
            note = "Strong result. The pipeline is performing well on this dimension."
        elif val >= 0.50:
            note = "Moderate result. Tune retriever or prompt."
        else:
            note = "Weak result. Fix retrieval quality or grounding."

        interp_html += f"""
        <div style="border-left:4px solid {col};padding:10px 16px;
                    margin-bottom:12px;background:#fafafa;border-radius:0 8px 8px 0">
          <span style="font-weight:600;color:{col}">{label} — {val:.4f}</span>
          <br><span style="font-size:13px;color:#555">{note}</span>
        </div>"""

    # ── Debug warning ──
    debug_warning = ""
    if nan_counter > 0:
        debug_warning = f"""
        <div style="background:#fff3cd;border:1px solid #ffeeba;
                    padding:12px 16px;border-radius:8px;margin-bottom:20px;
                    color:#856404;font-size:13px">
          ⚠ Detected <strong>{nan_counter}</strong> NaN metric values.
          This indicates evaluation instability (retriever issues or judge failure).
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>RAGAS Report</title>
</head>
<body style="font-family:sans-serif;background:#f5f5f5;padding:20px">

<h1>RAGAS Evaluation Report</h1>
<p>Generated: {now}</p>

{debug_warning}

<h2>Aggregate Scores</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap">
{cards_html}
</div>

<h2>Per Question</h2>
<table style="width:100%;border-collapse:collapse">
<thead>
<tr>
<th>#</th>
<th>Question</th>
<th>Answer</th>
<th>Faith</th>
<th>Rel</th>
<th>Prec</th>
<th>Rec</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Interpretation</h2>
{interp_html}

</body>
</html>"""

    return html


# ─────────────────────────────
# Runner
# ─────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ '{RESULTS_FILE}' not found.")
        sys.exit(1)

    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    html = generate_report(data)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report saved to: {REPORT_FILE}")