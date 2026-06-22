from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _report_date_text(value: str) -> str:
    text = str(value).strip()
    if text:
        return text
    return datetime.now(timezone.utc).date().isoformat()


def _comparison_row(label: str, report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return ""
    return (
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td class='num'>{html.escape(str(report.get('chrf', '')))}</td>"
        f"<td class='num'>{html.escape(str(report.get('bleu', '')))}</td>"
        f"<td class='num'>{html.escape(str(report.get('exact_match_pct', '')))}</td>"
        "</tr>"
    )


def build_html_report(
    *,
    artifact_dir: str,
    output_html: str,
    dataset_label: str,
    model_label: str,
    baseline_eval_json: str = "",
    preflight_json: str = "",
    report_date: str = "",
) -> dict[str, Any]:
    artifact_root = Path(artifact_dir).resolve()
    verified = _load_json(artifact_root / "verified_eval.json")
    train = _load_json(artifact_root / "train_metrics.json")
    samples = _load_json(artifact_root / "verified_samples.json")
    runtime_metadata = _optional_json(artifact_root / "runtime_metadata.json") or {}
    manifest_quality = _optional_json(artifact_root / "manifest_quality.json") or {}
    preflight = _optional_json(Path(preflight_json).resolve()) if str(preflight_json).strip() else None
    baseline = _optional_json(Path(baseline_eval_json).resolve()) if str(baseline_eval_json).strip() else None

    source_counts = {}
    if isinstance(preflight, dict):
        source_counts = preflight.get("aggregate_resolved_source_counts") or preflight.get("aggregate_manifest_source_counts") or {}
    if not source_counts:
        source_counts = verified.get("source_counts") or {}

    comparison_rows = "".join(
        row for row in [
            _comparison_row("Baseline", baseline),
            _comparison_row("Candidate", verified),
        ] if row
    )
    sample_rows = []
    for sample in list(samples)[:8]:
        sample_rows.append(
            "<tr>"
            f"<td>{html.escape(str(sample.get('reference', '')))}</td>"
            f"<td>{html.escape(str(sample.get('hypothesis', '')))}</td>"
            f"<td class='num'>{html.escape(str(sample.get('score', '')))}</td>"
            "</tr>"
        )

    html_text = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(model_label)} - Cloud Model Report</title>
<style>
body{{margin:0;background:#0f1117;color:#e7ecf3;font-family:"Segoe UI","Noto Sans Thai",system-ui,sans-serif;line-height:1.55}}
.wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}
.card{{background:#171b24;border:1px solid #2a3141;border-radius:12px;padding:18px}}
.stat{{font-size:30px;font-weight:700}}
table{{width:100%;border-collapse:collapse;background:#171b24;border:1px solid #2a3141;border-radius:12px;overflow:hidden}}
th,td{{padding:12px 14px;border-bottom:1px solid #2a3141;text-align:left;vertical-align:top}}
th{{font-size:12px;text-transform:uppercase;color:#9aa4b2}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
code{{background:#10141d;padding:2px 6px;border-radius:6px}}
.muted{{color:#9aa4b2}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(model_label)}</h1>
  <p class="muted">Cloud artifact: <code>{html.escape(str(artifact_root))}</code> | Dataset: <code>{html.escape(dataset_label)}</code> | Report date: {html.escape(_report_date_text(report_date))}</p>
  <div class="grid">
    <div class="card"><div class="muted">Readiness</div><div class="stat">{'Passed' if verified.get('chrf') is not None else 'Unknown'}</div></div>
    <div class="card"><div class="muted">chrF</div><div class="stat">{html.escape(str(verified.get('chrf', '')))}</div></div>
    <div class="card"><div class="muted">BLEU</div><div class="stat">{html.escape(str(verified.get('bleu', '')))}</div></div>
    <div class="card"><div class="muted">Exact Match %</div><div class="stat">{html.escape(str(verified.get('exact_match_pct', '')))}</div></div>
  </div>
  <h2>Dataset Gate</h2>
  <table>
    <tbody>
      <tr><th>Dataset label</th><td>{html.escape(dataset_label)}</td></tr>
      <tr><th>Source counts</th><td><code>{html.escape(json.dumps(source_counts, ensure_ascii=False, sort_keys=True))}</code></td></tr>
      <tr><th>Manifest quality passed</th><td>{html.escape(str(manifest_quality.get('passed', '')))}</td></tr>
      <tr><th>Cloud preflight</th><td>{html.escape(str((preflight or {}).get('passed', 'n/a')))}</td></tr>
      <tr><th>Resolved examples</th><td>{html.escape(str((preflight or {}).get('aggregate_resolved_examples', 'n/a')))}</td></tr>
    </tbody>
  </table>
  <h2>Runtime</h2>
  <table>
    <tbody>
      <tr><th>Base model</th><td>{html.escape(str(runtime_metadata.get('base_model', 'google/mt5-small')))}</td></tr>
      <tr><th>Export dir</th><td><code>{html.escape(str(runtime_metadata.get('export_dir', artifact_root)))}</code></td></tr>
      <tr><th>Checkpoint step</th><td>{html.escape(str(runtime_metadata.get('checkpoint_step', '')))}</td></tr>
      <tr><th>Stopped reason</th><td>{html.escape(str(train.get('stopped_reason', '')))}</td></tr>
    </tbody>
  </table>
  <h2>Comparison</h2>
  <table>
    <thead><tr><th>Run</th><th class="num">chrF</th><th class="num">BLEU</th><th class="num">Exact %</th></tr></thead>
    <tbody>{comparison_rows}</tbody>
  </table>
  <h2>Predictions</h2>
  <table>
    <thead><tr><th>Reference</th><th>Hypothesis</th><th class="num">Score</th></tr></thead>
    <tbody>{''.join(sample_rows)}</tbody>
  </table>
</div>
</body>
</html>
"""

    output_path = Path(output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "artifact_dir": str(artifact_root),
        "output_html": str(output_path),
        "dataset_label": dataset_label,
        "model_label": model_label,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an HTML model report from cloud PoseT5 artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--dataset-label", default="mixed_all_train_v6")
    parser.add_argument("--model-label", default="PoseToTextT5")
    parser.add_argument("--baseline-eval-json", default="")
    parser.add_argument("--preflight-json", default="")
    parser.add_argument("--report-date", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_html_report(
        artifact_dir=args.artifact_dir,
        output_html=args.output_html,
        dataset_label=args.dataset_label,
        model_label=args.model_label,
        baseline_eval_json=args.baseline_eval_json,
        preflight_json=args.preflight_json,
        report_date=args.report_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
