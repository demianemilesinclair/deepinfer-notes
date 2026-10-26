# -*- coding: utf-8 -*-
"""Build the 128-GPU cost model workbook (formulas live, edit the yellow cells)."""
import os

import xlsxwriter

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "deepinfer-128gpu-cost-model.xlsx")

wb = xlsxwriter.Workbook(OUT)
wb.set_properties({
    "title": "DeepInfer 128-GPU cost model",
    "subject": "DeepSeek V4.1 Flash minimum efficient deployment unit",
    "author": "Based on public DeepInfer numbers",
    "comments": "Estimates only. Edit the yellow cells and everything recomputes.",
})

title_f = wb.add_format({"bold": True, "font_size": 15, "font_color": "#1C1E24"})
sub_f = wb.add_format({"italic": True, "font_size": 10, "font_color": "#6E7480"})
hdr_f = wb.add_format({"bold": True, "bg_color": "#1F2937", "font_color": "white",
                       "align": "center", "valign": "vcenter", "border": 1})
sec_f = wb.add_format({"bold": True, "font_size": 12, "font_color": "#1D4ED8"})
lbl_f = wb.add_format({"border": 1, "bg_color": "#F3F4F6"})
in_f = wb.add_format({"border": 1, "bg_color": "#FEF3C7", "num_format": "#,##0.00"})
in_int_f = wb.add_format({"border": 1, "bg_color": "#FEF3C7", "num_format": "#,##0"})
out_f = wb.add_format({"border": 1, "num_format": "#,##0.00"})
out_int_f = wb.add_format({"border": 1, "num_format": "#,##0"})
usd_f = wb.add_format({"border": 1, "num_format": "$#,##0.00"})
usd4_f = wb.add_format({"border": 1, "num_format": "$#,##0.0000"})
pct_f = wb.add_format({"border": 1, "num_format": "0.0%"})
note_f = wb.add_format({"font_size": 9, "font_color": "#6E7480", "text_wrap": True, "valign": "top"})

# ---------------- Assumptions ----------------
ws = wb.add_worksheet("Assumptions")
ws.set_column("A:A", 38)
ws.set_column("B:B", 18)
ws.set_column("C:C", 64)

ws.write("A1", "DeepInfer 128-GPU cost model - assumptions", title_f)
ws.write("A2", "Yellow cells are editable; every other sheet recomputes. Numbers are public data plus the estimates in the write-up.", sub_f)

rows = [
    ("Total GPUs", 128, "64 prefill + 64 decode: the minimum efficient unit from the official Q&A"),
    ("Prefill GPUs", 64, ""),
    ("Decode GPUs", 64, ""),
    ("Rental price (USD / GPU-hour)", 2.5, "B200-class spot runs 2-3, on-demand 4-6; replace with your own quote"),
    ("Decode throughput per GPU (tokens/s)", 14466, "Official benchmark: EP64/DP64, 60 concurrent sequences per rank"),
    ("Hours per day", 24, ""),
    ("Days per month", 30, ""),
    ("Official revenue benchmark (USD / GPU-hour)", 38, "Official Q&A: revenue per GPU-hour at off-peak pricing (revenue, not profit)"),
    ("Official cluster volume (tokens / day)", 155e12, "155 trillion tokens per day"),
    ("KV cache per GPU (TB, logical)", 14.4, "Official: about 14.4 TB per GPU, logical"),
    ("3FS erasure-coding overhead", 1.2, "10+2 erasure coding: physical = logical x 1.2"),
    ("Usable capacity per storage server (TB)", 614.6, "20 x 30.73 TB SSDs"),
]
r = 4
pos = {}
for name, val, note in rows:
    ws.write(r, 0, name, lbl_f)
    if isinstance(val, float) and val > 1e6:
        ws.write_number(r, 1, val, in_int_f)
    elif isinstance(val, float) and not float(val).is_integer():
        ws.write_number(r, 1, val, in_f)
    else:
        ws.write_number(r, 1, val, in_int_f)
    ws.write(r, 2, note, note_f)
    pos[name] = f"Assumptions!$B${r + 1}"
    r += 1

price = pos["Rental price (USD / GPU-hour)"]
gpus = pos["Total GPUs"]
tps = pos["Decode throughput per GPU (tokens/s)"]
hours = pos["Hours per day"]
days = pos["Days per month"]
rev = pos["Official revenue benchmark (USD / GPU-hour)"]
volume = pos["Official cluster volume (tokens / day)"]
kv = pos["KV cache per GPU (TB, logical)"]
ec = pos["3FS erasure-coding overhead"]
srv = pos["Usable capacity per storage server (TB)"]

# ---------------- Cost model ----------------
ws2 = wb.add_worksheet("Cost model")
ws2.set_column("A:A", 36)
ws2.set_column("B:F", 16)
ws2.set_column("D:D", 56)

ws2.write("A1", "Cost model", title_f)
ws2.write("A2", "Change the yellow cells on the Assumptions sheet; this sheet follows.", sub_f)

ws2.write("A4", "Throughput and cost", sec_f)
for col, head in enumerate(["Metric", "Value", "Unit", "Notes"]):
    ws2.write(4, col, head, hdr_f)

r = 5
ws2.write(r, 0, "Total throughput (full load)", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{tps}", out_int_f)
ws2.write(r, 2, "tokens/s", lbl_f)
ws2.write(r, 3, "GPU count x per-GPU throughput", note_f)
r += 1
ws2.write(r, 0, "Daily output (full load)", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{tps}*{hours}*3600", out_int_f)
ws2.write(r, 2, "tokens/day", lbl_f)
ws2.write(r, 3, "about 160 billion", note_f)
r += 1
ws2.write(r, 0, "Cost per million tokens (full load)", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{price}/({gpus}*{tps}*3600/1000000)", usd4_f)
ws2.write(r, 2, "$/M tokens", lbl_f)
ws2.write(r, 3, "= 128 x price / tokens per hour", note_f)
r += 1
ws2.write(r, 0, "Rental cost per hour", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{price}", usd_f)
ws2.write(r, 2, "USD/hour", lbl_f)
ws2.write(r, 3, "", note_f)
r += 1
ws2.write(r, 0, "Rental cost per day", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{price}*{hours}", usd_f)
ws2.write(r, 2, "USD/day", lbl_f)
ws2.write(r, 3, "", note_f)
r += 1
ws2.write(r, 0, "Rental cost per month", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{price}*{hours}*{days}", usd_f)
ws2.write(r, 2, "USD/month", lbl_f)
ws2.write(r, 3, "excludes storage, network and staff", note_f)
r += 1
ws2.write(r, 0, "One-week trial", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{price}*{hours}*7", usd_f)
ws2.write(r, 2, "USD", lbl_f)
ws2.write(r, 3, "the entry ticket for a single validation run", note_f)
r += 2

ws2.write(r, 0, "Derived from the official revenue benchmark", sec_f)
r += 1
ws2.write(r, 0, "Revenue at full load (128 GPUs)", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{rev}", usd_f)
ws2.write(r, 2, "USD/hour", lbl_f)
ws2.write(r, 3, "official figure: 38 USD per GPU-hour", note_f)
r += 1
ws2.write(r, 0, "Implied price", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{rev}/({gpus}*{tps}*3600/1000000)", usd4_f)
ws2.write(r, 2, "$/M tokens", lbl_f)
ws2.write(r, 3, "blended; no input/output split, excludes storage", note_f)
r += 1
ws2.write(r, 0, "Break-even utilisation", lbl_f)
ws2.write_formula(r, 1, f"={price}/{rev}", pct_f)
ws2.write(r, 2, "", lbl_f)
ws2.write(r, 3, "= rental price / 38; below this you lose money", note_f)
r += 2

ws2.write(r, 0, "Utilisation sensitivity (cost per million tokens)", sec_f)
r += 1
for col, head in enumerate(["Utilisation", "Cost $/M", "Daily output (tokens)"]):
    ws2.write(r, col, head, hdr_f)
for u in (1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.066):
    r += 1
    ws2.write_number(r, 0, u, pct_f)
    ws2.write_formula(r, 1, f"={gpus}*{price}/({gpus}*{tps}*3600*{u}/1000000)", usd4_f)
    ws2.write_formula(r, 2, f"={gpus}*{tps}*{hours}*3600*{u}", out_int_f)
r += 2

ws2.write(r, 0, "Storage and scale", sec_f)
r += 1
ws2.write(r, 0, "KV cache, logical", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{kv}/1000", out_f)
ws2.write(r, 2, "PB", lbl_f)
ws2.write(r, 3, "14.4 TB per GPU x GPU count", note_f)
r += 1
ws2.write(r, 0, "KV cache, physical", lbl_f)
ws2.write_formula(r, 1, f"={gpus}*{kv}/1000*{ec}", out_f)
ws2.write(r, 2, "PB", lbl_f)
ws2.write(r, 3, "after 10+2 erasure coding", note_f)
r += 1
ws2.write(r, 0, "Storage servers", lbl_f)
ws2.write_formula(r, 1, f"=ROUNDUP({gpus}*{kv}/1000*{ec}*1000/{srv},0)", out_int_f)
ws2.write(r, 2, "servers", lbl_f)
ws2.write(r, 3, "each with 20 x 30.73 TB", note_f)
r += 1
ws2.write(r, 0, "Equivalent 128-GPU units", lbl_f)
ws2.write_formula(r, 1, f"={volume}/({gpus}*{tps}*{hours}*3600)", out_f)
ws2.write(r, 2, "units", lbl_f)
ws2.write(r, 3, "155T/day divided by one unit's daily output; arithmetic only", note_f)
r += 1
ws2.write(r, 0, "Implied total GPUs", lbl_f)
ws2.write_formula(r, 1, f"={volume}/({gpus}*{tps}*{hours}*3600)*{gpus}", out_int_f)
ws2.write(r, 2, "GPUs", lbl_f)
ws2.write(r, 3, "about 120,000; derived, not disclosed", note_f)

ws2.merge_range(r + 2, 0, r + 3, 3, "", note_f)
ws2.write(r + 2, 0,
          "Estimates only, for order-of-magnitude reasoning, not procurement. Throughput comes from the official "
          "benchmark at 60 concurrent sequences per rank; a real deployment loses some of it to prefill, cache "
          "misses and request long tails.", note_f)

wb.close()
print("saved", OUT)
