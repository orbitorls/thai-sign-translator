"""Promote a verified PoseT5 export only when it beats the incumbent."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a candidate PoseT5 export only when its eval beats the incumbent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-export-dir", required=True)
    parser.add_argument("--candidate-eval-json", required=True)
    parser.add_argument("--stable-export-dir", required=True)
    parser.add_argument("--stable-eval-json", required=True)
    parser.add_argument("--candidate-samples-json", default=None)
    parser.add_argument("--stable-samples-json", default=None)
    parser.add_argument("--min-source-examples", type=int, default=5)
    parser.add_argument("--min-source-chrf", type=float, default=20.0)
    parser.add_argument("--min-source-exact-match-pct", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _candidate_sort_key(report: dict) -> tuple[float, float, float]:
    return (
        float(report.get("chrf", 0.0)),
        float(report.get("bleu", 0.0)),
        float(report.get("exact_match_pct", 0.0)),
    )


def _source_reports(report: dict | None) -> dict[str, dict]:
    if not isinstance(report, dict):
        return {}
    source_metrics = report.get("source_metrics")
    return source_metrics if isinstance(source_metrics, dict) else {}


def _report_data_roots(report: dict | None) -> list[str]:
    if not isinstance(report, dict):
        return []
    data_roots = report.get("data_roots")
    if not isinstance(data_roots, list):
        return []
    return [str(root) for root in data_roots if str(root).strip()]


def _report_source_names(report: dict | None) -> list[str]:
    source_reports = _source_reports(report)
    if source_reports:
        return sorted(str(source) for source in source_reports)

    if not isinstance(report, dict):
        return []
    source_counts = report.get("source_counts")
    if not isinstance(source_counts, dict):
        return []
    return sorted(str(source) for source in source_counts)


def _is_clearly_single_source(report: dict | None) -> bool:
    source_names = _report_source_names(report)
    return len(source_names) == 1


def _needs_source_gates(candidate_report: dict, incumbent_report: dict | None) -> bool:
    if incumbent_report is None:
        return not _is_clearly_single_source(candidate_report)
    return not (
        _is_clearly_single_source(candidate_report)
        and _is_clearly_single_source(incumbent_report)
    )


def _validate_eval_compatibility(candidate_report: dict, incumbent_report: dict) -> tuple[bool, str | None]:
    for field in ("seed", "val_subset_size", "data_roots"):
        if candidate_report.get(field) != incumbent_report.get(field):
            return False, f"eval mismatch on {field}"
    return True, None


def _validate_source_requirements(
    candidate_report: dict,
    incumbent_report: dict | None,
    min_source_examples: int,
    min_source_chrf: float,
    min_source_exact_match_pct: float,
) -> tuple[bool, str | None]:
    candidate_sources = _source_reports(candidate_report)
    if incumbent_report is None and _is_clearly_single_source(candidate_report) and len(candidate_sources) <= 1:
        return True, None
    if not candidate_sources:
        return False, "candidate missing source_metrics"

    for source in sorted(candidate_sources):
        metrics = candidate_sources[source]
        n = int(metrics.get("n", 0))
        chrf = float(metrics.get("chrf", 0.0))
        exact_match_pct = float(metrics.get("exact_match_pct", 0.0))
        if n < min_source_examples:
            return False, f"source {source} n {n} < required {min_source_examples}"
        if chrf < min_source_chrf:
            return False, f"source {source} chrf {chrf:.2f} < required {min_source_chrf:.2f}"
        if exact_match_pct < min_source_exact_match_pct:
            return False, (
                f"source {source} exact_match_pct {exact_match_pct:.2f} < required "
                f"{min_source_exact_match_pct:.2f}"
            )

    if incumbent_report is None:
        return True, None

    incumbent_sources = _source_reports(incumbent_report)
    if not incumbent_sources:
        return False, "incumbent missing source_metrics"
    if sorted(candidate_sources) != sorted(incumbent_sources):
        return False, "candidate and incumbent source sets differ"

    for source in sorted(incumbent_sources):
        candidate_metrics = candidate_sources[source]
        incumbent_metrics = incumbent_sources[source]
        if int(candidate_metrics.get("n", 0)) != int(incumbent_metrics.get("n", 0)):
            return False, f"source {source} n differs from incumbent"
        if float(candidate_metrics.get("chrf", 0.0)) < float(incumbent_metrics.get("chrf", 0.0)):
            return False, f"source {source} regressed on chrf"
        if float(candidate_metrics.get("exact_match_pct", 0.0)) < float(incumbent_metrics.get("exact_match_pct", 0.0)):
            return False, f"source {source} regressed on exact_match_pct"
    return True, None


def _copy_tree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _rewrite_export_paths(stable_export_dir: Path, stable_eval_json: Path) -> None:
    runtime_metadata_path = stable_export_dir / "runtime_metadata.json"
    if runtime_metadata_path.is_file():
        data = _load_json(runtime_metadata_path)
        data["export_dir"] = str(stable_export_dir)
        runtime_metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if stable_eval_json.is_file():
        data = _load_json(stable_eval_json)
        data["export_dir"] = str(stable_export_dir)
        runtime_metadata = data.get("runtime_metadata")
        if isinstance(runtime_metadata, dict):
            runtime_metadata["export_dir"] = str(stable_export_dir)
        stable_eval_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _promote(args: argparse.Namespace) -> dict:
    candidate_export_dir = Path(args.candidate_export_dir).resolve()
    candidate_eval_json = Path(args.candidate_eval_json).resolve()
    stable_export_dir = Path(args.stable_export_dir).resolve()
    stable_eval_json = Path(args.stable_eval_json).resolve()
    candidate_samples_json = Path(args.candidate_samples_json).resolve() if args.candidate_samples_json else None
    stable_samples_json = Path(args.stable_samples_json).resolve() if args.stable_samples_json else None

    candidate_report = _load_json(candidate_eval_json)
    incumbent_report = _load_json(stable_eval_json) if stable_eval_json.is_file() else None
    candidate_readiness = candidate_report.get("promotion_status")
    if (
        not args.force
        and isinstance(candidate_readiness, dict)
        and candidate_readiness.get("ready") is False
    ):
        failures = candidate_readiness.get("failures")
        reason = "candidate promotion_status.ready is false"
        if isinstance(failures, list) and failures:
            reason = f"{reason}: {'; '.join(str(item) for item in failures)}"
        return {
            "promoted": False,
            "reason": reason,
            "candidate_export_dir": str(candidate_export_dir),
            "stable_export_dir": str(stable_export_dir),
            "candidate_metrics": _candidate_sort_key(candidate_report),
            "incumbent_metrics": _candidate_sort_key(incumbent_report) if incumbent_report is not None else None,
            "candidate_source_metrics": _source_reports(candidate_report),
            "incumbent_source_metrics": _source_reports(incumbent_report),
        }

    promoted = False
    reason = "candidate did not beat incumbent"
    if args.force:
        promoted = True
        reason = "forced"
    elif incumbent_report is None:
        source_ok, source_reason = _validate_source_requirements(
            candidate_report,
            None,
            args.min_source_examples,
            args.min_source_chrf,
            args.min_source_exact_match_pct,
        )
        if source_ok:
            promoted = True
            reason = "no incumbent"
        else:
            reason = source_reason or reason
    elif _candidate_sort_key(candidate_report) > _candidate_sort_key(incumbent_report):
        if _needs_source_gates(candidate_report, incumbent_report):
            comparable, comparable_reason = _validate_eval_compatibility(candidate_report, incumbent_report)
            if not comparable:
                reason = comparable_reason or reason
            else:
                source_ok, source_reason = _validate_source_requirements(
                    candidate_report,
                    incumbent_report,
                    args.min_source_examples,
                    args.min_source_chrf,
                    args.min_source_exact_match_pct,
                )
                if source_ok:
                    promoted = True
                    reason = "candidate beat incumbent"
                else:
                    reason = source_reason or reason
        else:
            promoted = True
            reason = "candidate beat incumbent"

    if promoted:
        _copy_tree_replace(candidate_export_dir, stable_export_dir)
        stable_eval_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_eval_json, stable_eval_json)
        if candidate_samples_json and stable_samples_json:
            shutil.copy2(candidate_samples_json, stable_samples_json)
        _rewrite_export_paths(stable_export_dir, stable_eval_json)

    return {
        "promoted": promoted,
        "reason": reason,
        "candidate_export_dir": str(candidate_export_dir),
        "stable_export_dir": str(stable_export_dir),
        "candidate_metrics": _candidate_sort_key(candidate_report),
        "incumbent_metrics": _candidate_sort_key(incumbent_report) if incumbent_report is not None else None,
        "candidate_source_metrics": _source_reports(candidate_report),
        "incumbent_source_metrics": _source_reports(incumbent_report),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = _promote(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
