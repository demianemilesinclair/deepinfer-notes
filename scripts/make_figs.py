# -*- coding: utf-8 -*-
"""Render the async-scheduling diagram (sync stepping vs async scheduling) with Pillow."""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1800, 1000
BG = (255, 255, 255)
INK = (28, 30, 36)
MUTED = (110, 116, 128)
GRID = (228, 231, 236)
GPU_C = (37, 99, 235)
GPU_C2 = (96, 165, 250)
CPU_C = (16, 145, 110)
IDLE = (234, 237, 242)
BUBBLE = (220, 55, 55)

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"


def f(size, bold=False):
    path = FONT_B if bold and os.path.exists(FONT_B) else FONT
    return ImageFont.truetype(path, size)


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

x0, x1 = 300, 1720
UNIT = (x1 - x0) / 13.0


def bx(t):
    return x0 + t * UNIT


def block(y, h, t0, t1, fill, label, font, fg=(255, 255, 255), radius=8, min_font=11):
    x_a, x_b = bx(t0), bx(t1)
    d.rounded_rectangle([x_a, y, x_b, y + h], radius=radius, fill=fill)
    size = font.size
    while size > min_font and d.textlength(label, font=f(size)) > (x_b - x_a) - 14:
        size -= 1
    ft = f(size)
    tw = d.textlength(label, font=ft)
    d.text((x_a + ((x_b - x_a) - tw) / 2, y + (h - size) / 2 - 2), label, font=ft, fill=fg)


d.text((60, 40), "Why async scheduling removes the GPU bubble", font=f(34, True), fill=INK)
d.text((60, 92), "Wall-clock time across decode steps: naive synchronous stepping above, DeepInfer's zero-overhead batch scheduler below",
       font=f(18), fill=MUTED)

LBL = f(19)
SM = f(16)
TAG = f(18, True)


def panel(y_top, title, note):
    d.text((60, y_top), title, font=f(24, True), fill=INK)
    d.text((60, y_top + 38), note, font=SM, fill=MUTED)


# ---------------- panel A ----------------
A = 180
panel(A, "(1) Naive synchronous stepping: the GPU waits for the CPU",
      "Each step must wait for sample(N) before post-processing and scheduling the next batch, so the GPU idles")
g1, c1, bh = A + 96, A + 224, 78
d.text((60, g1 + 26), "GPU", font=TAG, fill=GPU_C)
d.text((60, c1 + 26), "CPU", font=TAG, fill=CPU_C)

block(g1, bh, 0, 3, GPU_C, "forward(N)", LBL)
block(g1, bh, 3, 4, GPU_C2, "sample(N)", LBL)
block(g1, bh, 4, 6.5, IDLE, "bubble: waiting for CPU", SM, fg=BUBBLE)
block(g1, bh, 6.5, 9.5, GPU_C, "forward(N+1)", LBL)
block(g1, bh, 9.5, 10.5, GPU_C2, "sample(N+1)", LBL)
block(g1, bh, 10.5, 13, IDLE, "bubble", SM, fg=BUBBLE)

block(c1, bh, 4, 5.2, CPU_C, "post(N)", SM)
block(c1, bh, 5.2, 6.5, CPU_C, "schedule(N+1)", SM)

d.text((bx(4) + 4, g1 + bh + 10), "this idle time is a direct throughput loss", font=SM, fill=BUBBLE)

# ---------------- panel B ----------------
B = 540
panel(B, "(2) Async scheduling: CPU work hides inside the previous step's GPU time",
      "The GPU runs back to back; the CPU computes the next budget early and defers resource release by one step")
g2, c2 = B + 96, B + 224
d.text((60, g2 + 26), "GPU", font=TAG, fill=GPU_C)
d.text((60, c2 + 26), "CPU", font=TAG, fill=CPU_C)

block(g2, bh, 0, 3, GPU_C, "forward(N)", LBL)
block(g2, bh, 3, 4, GPU_C2, "sample(N)", LBL)
block(g2, bh, 4, 7, GPU_C, "forward(N+1)", LBL)
block(g2, bh, 7, 8, GPU_C2, "sample(N+1)", LBL)
block(g2, bh, 8, 11, GPU_C, "forward(N+2)", LBL)
block(g2, bh, 11, 12, GPU_C2, "sample", LBL)

block(c2, bh, 0, 1.45, CPU_C, "schedule(N+1)", SM)
block(c2, bh, 1.6, 3.05, CPU_C, "post(N-1)", SM)
block(c2, bh, 3.2, 4.65, CPU_C, "schedule(N+2)", SM)
block(c2, bh, 4.8, 6.25, CPU_C, "post(N)", SM)
block(c2, bh, 6.4, 7.85, CPU_C, "schedule(N+3)", SM)
block(c2, bh, 8.0, 9.45, CPU_C, "post(N+1)", SM)

d.text((bx(4) + 4, g2 + bh + 10), "three forwards run back to back; the bubbles are gone", font=SM, fill=CPU_C)

axis_y = B + 350
d.line([x0, axis_y, x1, axis_y], fill=GRID, width=2)
for t in range(0, 14):
    d.line([bx(t), axis_y - 6, bx(t), axis_y + 6], fill=GRID, width=2)
d.text((bx(0) - 8, axis_y - 32), "t0", font=SM, fill=MUTED)
d.text((x1 - 60, axis_y - 32), "time", font=SM, fill=MUTED)

ly = H - 86
d.rounded_rectangle([60, ly + 5, 84, ly + 29], radius=5, fill=GPU_C)
d.text((94, ly + 5), "GPU forward", font=SM, fill=INK)
d.rounded_rectangle([240, ly + 5, 264, ly + 29], radius=5, fill=GPU_C2)
d.text((274, ly + 5), "GPU sample", font=SM, fill=INK)
d.rounded_rectangle([400, ly + 5, 424, ly + 29], radius=5, fill=CPU_C)
d.text((434, ly + 5), "CPU logic (schedule / post-process / resource release)", font=SM, fill=INK)
d.rounded_rectangle([940, ly + 5, 964, ly + 29], radius=5, fill=IDLE)
d.text((974, ly + 5), "GPU idle (bubble)", font=SM, fill=INK)

d.text((60, H - 36), "Source: DeepInfer doc/inference_deployment_qa.md, section \"Asynchronous scheduling\"; schematic, not a measured trace",
       font=f(14), fill=MUTED)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs", "async_pipeline.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out, "PNG")
print("saved", out, img.size)
