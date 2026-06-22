from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "tmp" / "kaggle_kernel_output" / "pose_t5_tsl51_tune3"
OUTPUT_PATH = REPO_ROOT / "docs" / "model_report.png"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/LeelaUIb.ttf",
                "C:/Windows/Fonts/leelawdb.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/tahomabd.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/LeelawUI.ttf",
            "C:/Windows/Fonts/leelawad.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _rounded(draw: ImageDraw.ImageDraw, box, fill, outline):
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)


def _text(draw: ImageDraw.ImageDraw, xy, text, *, font, fill, anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def main() -> int:
    verified = _load_json(ARTIFACT_DIR / "verified_eval.json")
    sampling = _load_json(ARTIFACT_DIR / "source_sampling.json")
    train = _load_json(ARTIFACT_DIR / "train_metrics.json")
    samples = _load_json(ARTIFACT_DIR / "verified_samples.json")

    width, height = 1600, 960
    image = Image.new("RGB", (width, height), "#0f1117")
    draw = ImageDraw.Draw(image)

    bg2 = "#171b24"
    line = "#2a3141"
    text = "#e7ecf3"
    muted = "#9aa4b2"
    blue = "#58a6ff"
    green = "#3fb950"
    amber = "#d29922"
    red = "#f85149"

    title_font = _font(44, bold=True)
    h_font = _font(24, bold=True)
    body_font = _font(19)
    small_font = _font(16)
    stat_font = _font(38, bold=True)
    badge_font = _font(15, bold=True)

    draw.rectangle((0, 0, width, height), fill="#0f1117")
    _text(draw, (52, 42), "MODEL REPORT", font=badge_font, fill=blue)
    _text(draw, (52, 78), "Thai Sign Translator - Latest Cloud-Verified Model", font=title_font, fill=text)
    _text(
        draw,
        (52, 136),
        "Cloud artifact: tmp/kaggle_kernel_output/pose_t5_tsl51_tune3 | Report date: 2026-06-21",
        font=body_font,
        fill=muted,
    )

    cards = [
        (52, 182, 385, 318, "Readiness", "Passed", "promotion_status.ready = true", green),
        (405, 182, 738, 318, "chrF", f"{verified['chrf']}", "BLEU 94.79", blue),
        (758, 182, 1091, 318, "Exact Match", f"{verified['exact_match_pct']}%", "20 / 25 examples", green),
        (1111, 182, 1548, 318, "Resume Delta", f"+{train['new_optimizer_steps']}", "steps from 3075 to 3400", amber),
    ]
    for x1, y1, x2, y2, label, value, hint, accent in cards:
        _rounded(draw, (x1, y1, x2, y2), bg2, line)
        _text(draw, (x1 + 22, y1 + 20), label, font=body_font, fill=muted)
        _text(draw, (x1 + 22, y1 + 68), value, font=stat_font, fill=text)
        _text(draw, (x1 + 22, y2 - 32), hint, font=small_font, fill=accent)

    _rounded(draw, (52, 344, 760, 612), bg2, line)
    _text(draw, (74, 366), "Comparison", font=h_font, fill=text)
    headers = ["Run", "chrF", "BLEU", "Exact %"]
    col_x = [74, 430, 545, 665]
    for idx, header in enumerate(headers):
        _text(draw, (col_x[idx], 408), header, font=small_font, fill=muted)
    rows = [
        ("Incumbent baseline", "86.95", "89.38", "64.0"),
        ("Previous cloud run", "90.37", "91.82", "72.0"),
        ("Latest cloud run", "94.4", "94.79", "80.0"),
    ]
    y = 450
    for i, row in enumerate(rows):
        row_fill = "#1c2230" if i == 2 else "#141922"
        draw.rounded_rectangle((68, y - 12, 742, y + 34), radius=12, fill=row_fill)
        for idx, cell in enumerate(row):
            fill = green if (i == 2 and idx > 0) else text
            _text(draw, (col_x[idx], y), cell, font=body_font, fill=fill)
        y += 62

    _rounded(draw, (784, 344, 1548, 612), bg2, line)
    _text(draw, (806, 366), "Training Delta That Cleared The Gate", font=h_font, fill=text)
    train_lines = [
        "resume step: 3075",
        "final step: 3400",
        "stop reason: early_stopping",
        "best resumed chrF: 94.4040",
        "best val loss: 0.1049",
        "latest verified decode: beam=5, max_new_tokens=72",
    ]
    y = 414
    for line_text in train_lines:
        _text(draw, (812, y), line_text, font=body_font, fill=text)
        y += 34

    _rounded(draw, (52, 638, 760, 876), bg2, line)
    _text(draw, (74, 660), "Focus-token weighting", font=h_font, fill=text)
    focus_tokens = [
        ("ฉัน", "1.0x"),
        ("คุณ", "1.96x"),
        ("แม่", "3.0x"),
        ("พี่", "3.0x"),
        ("วันนี้", "3.0x"),
        ("พรุ่งนี้", "3.0x"),
    ]
    y = 706
    for token, multiplier in focus_tokens:
        _text(draw, (82, y), token, font=body_font, fill=text)
        draw.rounded_rectangle((220, y + 4, 420, y + 22), radius=9, fill="#10141d")
        fill_width = 60 if multiplier == "1.0x" else (120 if multiplier == "1.96x" else 180)
        draw.rounded_rectangle((220, y + 4, 220 + fill_width, y + 22), radius=9, fill=green)
        _text(draw, (438, y), multiplier, font=body_font, fill=text)
        y += 28
    _text(
        draw,
        (82, 848),
        f"focus examples: {sampling['focus_examples']} / {sum(sampling['source_counts'].values())}",
        font=body_font,
        fill=muted,
    )

    _rounded(draw, (784, 638, 1548, 876), bg2, line)
    _text(draw, (806, 660), "Residual Errors", font=h_font, fill=text)
    residuals = [
        "แม่/พี่ -> sample miss still present",
        "ฉัน/คุณ -> sample miss still present",
        "วันนี้/พรุ่งนี้ -> time token confusion remains",
    ]
    colors = [red, amber, amber]
    y = 712
    for idx, item in enumerate(residuals):
        draw.rounded_rectangle((810, y - 8, 1516, y + 26), radius=12, fill="#141922")
        _text(draw, (826, y), item, font=body_font, fill=colors[idx])
        y += 48

    sample_text = "Example wins: " + " | ".join(
        [
            sample["reference"]
            for sample in samples
            if sample["reference"] == sample["hypothesis"]
        ][:2]
    )
    _text(draw, (826, 854), sample_text, font=small_font, fill=muted)

    footer = (
        "Artifacts: verified_eval.json, verified_samples.json, source_sampling.json, "
        "train_metrics.json, manifest_quality.json"
    )
    _text(draw, (52, 924), footer, font=small_font, fill=muted)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
