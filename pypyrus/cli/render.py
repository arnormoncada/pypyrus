from __future__ import annotations

import json
from typing import Any


def format_json(data: Any) -> str:
    """Render structured CLI output as pretty JSON."""
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def render_runs_table(runs: list[dict[str, Any]]) -> str:
    """Render a compact runs table for terminal output."""
    if not runs:
        return "No runs found."

    headers = (
        "RUN ID",
        "NAME",
        "STATUS",
        "START",
        "DURATION",
        "DATASETS",
        "LOADERS",
        "ROLES",
        "BATCHES",
    )
    rows = [
        (
            str(run.get("run_id") or ""),
            str(run.get("run_name") or ""),
            str(run.get("status") or "active"),
            str(run.get("start_time") or ""),
            _format_duration(run.get("duration_seconds")),
            str(run.get("dataset_count") or 0),
            str(run.get("loader_count") or 0),
            ",".join(run.get("roles") or []) or "-",
            str(run.get("batch_count") or 0),
        )
        for run in runs
    ]
    return _render_table(headers, rows)


def render_run_overview(overview: dict[str, Any]) -> str:
    """Render a human-readable run overview."""
    run = overview["run"]
    datasets = overview["datasets"]
    loaders = overview["loaders"]
    transforms = overview["transforms"]
    environment = overview["environment"]
    batches_by_role = overview["batches_by_role"]

    lines = [
        "Run overview",
        "-" * 60,
        f"Run ID: {run.get('run_id')}",
    ]
    if run.get("run_name"):
        lines.append(f"Run name: {run.get('run_name')}")
    lines.extend(
        [
            f"Status: {run.get('status') or 'active'}",
            f"Start: {run.get('start_time')}",
            f"End: {run.get('end_time') or '<active>'}",
            f"Duration: {_format_duration(run.get('duration_seconds'))}",
            f"Code ref: {run.get('code_ref') or '<none>'}",
            f"Config ref: {run.get('config_ref') or '<none>'}",
            f"Config json: {_format_config_json(run.get('config_json'))}",
        ]
    )
    seed_summary = _format_seed_summary(run.get("seed_summary_json"))
    if seed_summary != "<none>":
        lines.append(f"Seed summary: {seed_summary}")
    lines.extend(
        [
            "",
            "Summary",
            f"  Datasets: {len(datasets)}",
            f"  Loaders: {len(loaders)}",
            f"  Transforms: {len(transforms)}",
            f"  Batches: {overview['batch_count']}",
        ]
    )

    if batches_by_role:
        role_counts = ", ".join(
            f"{role}={count}" for role, count in sorted(batches_by_role.items())
        )
        lines.append(f"  Batch counts by role: {role_counts}")

    if datasets:
        lines.append("")
        lines.append("Datasets")
        for dataset in datasets:
            lines.append(f"  [{dataset.get('role')}] {dataset.get('name')}")
            lines.append(f"    dataset_id: {dataset.get('dataset_id')}")
            lines.append(f"    fingerprint: {dataset.get('fingerprint') or '<none>'}")
            lines.append(
                f"    fingerprint_method: {dataset.get('fingerprint_method') or '<none>'}"
            )
            lines.append(
                f"    sample_id_scheme: {dataset.get('sample_id_scheme') or '<none>'}"
            )
            lines.append(
                f"    sample_id_resolver: {dataset.get('sample_id_resolver') or '<none>'}"
            )
            if dataset.get("uri"):
                lines.append(f"    uri: {dataset.get('uri')}")
            if dataset.get("version_hint"):
                lines.append(f"    version_hint: {dataset.get('version_hint')}")

    if loaders:
        lines.append("")
        lines.append("Loaders")
        for loader in loaders:
            lines.append(
                "  "
                f"[{loader.get('role')}] "
                f"{loader.get('loader_id')} "
                f"(dataset_id={loader.get('dataset_id')})"
            )

    if transforms:
        lines.append("")
        lines.append("Transforms")
        for transform in transforms:
            transform_list = transform.get("transform_list") or []
            transform_names = ", ".join(item.get("name", "?") for item in transform_list)
            lines.append(f"  dataset_id={transform.get('dataset_id')}")
            lines.append(f"    names: {transform_names or '<none>'}")
            lines.append(
                f"    introspection_level: {transform.get('introspection_level')}"
            )
            lines.append(f"    params_hash: {transform.get('params_hash')}")

    if environment:
        snapshot = environment[0]
        lines.append("")
        lines.append("Environment")
        lines.append(f"  Python: {snapshot.get('python_version')}")
        lines.append(f"  CUDA: {snapshot.get('cuda_version') or '<none>'}")
        lines.append(
            "  "
            f"Library versions hash: {snapshot.get('library_versions_hash') or '<none>'}"
        )
        lines.append(
            f"  Hardware summary: {snapshot.get('hardware_summary') or '<none>'}"
        )

    return "\n".join(lines)


def render_batch(batch: dict[str, Any]) -> str:
    """Render a single batch record."""
    sample_ids = batch.get("sample_ids")
    sample_ids_text = (
        f"[{', '.join(map(str, sample_ids))}]"
        if sample_ids is not None
        else "<not stored>"
    )
    lines = [
        "Batch",
        "-" * 60,
        f"Run ID: {batch.get('run_id')}",
        f"Role: {batch.get('role')}",
        f"Loader ID: {batch.get('loader_id')}",
        f"Dataset ID: {batch.get('dataset_id')}",
        f"Timestamp: {batch.get('timestamp') or '<unknown>'}",
        f"Global step: {batch.get('global_step')}",
        f"Global sequence: {batch.get('global_sequence')}",
        f"Batch size: {batch.get('batch_size')}",
        f"Batch fingerprint: {batch.get('batch_fingerprint')}",
        f"Sample IDs: {sample_ids_text}",
    ]
    return "\n".join(lines)


def render_sample_find(result: dict[str, Any]) -> str:
    lines = [
        "Sample lookup",
        "-" * 60,
        f"Run ID: {result.get('run_id')}",
        f"Sample ID: {result.get('sample_id')}",
        f"Sample ID scheme: {result.get('sample_id_scheme')}",
        f"Found: {result.get('found')}",
        f"Occurrences: {result.get('occurrence_count')}",
        f"Matching steps: {result.get('matching_steps') or []}",
    ]

    if result.get("matching_roles"):
        lines.append(f"Roles: {', '.join(result['matching_roles'])}")
    if result.get("matching_loader_ids"):
        lines.append(f"Loaders: {', '.join(result['matching_loader_ids'])}")
    if result.get("matching_dataset_ids"):
        lines.append(f"Datasets: {', '.join(result['matching_dataset_ids'])}")

    first_occurrence = result.get("first_occurrence")
    if first_occurrence is not None:
        lines.append("")
        lines.append("First occurrence")
        lines.append(f"  Global sequence: {first_occurrence.get('global_sequence')}")
        lines.append(f"  Role: {first_occurrence.get('role')}")
        lines.append(f"  Loader ID: {first_occurrence.get('loader_id')}")
        lines.append(f"  Dataset ID: {first_occurrence.get('dataset_id')}")

    last_occurrence = result.get("last_occurrence")
    if last_occurrence is not None:
        lines.append("")
        lines.append("Last occurrence")
        lines.append(f"  Global sequence: {last_occurrence.get('global_sequence')}")
        lines.append(f"  Role: {last_occurrence.get('role')}")
        lines.append(f"  Loader ID: {last_occurrence.get('loader_id')}")
        lines.append(f"  Dataset ID: {last_occurrence.get('dataset_id')}")

    return "\n".join(lines)


def _render_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [
        render_row(headers),
        render_row(tuple("-" * width for width in widths)),
    ]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _format_duration(duration_seconds: Any) -> str:
    if duration_seconds is None:
        return "<active>"

    try:
        total_seconds = int(round(float(duration_seconds)))
    except (TypeError, ValueError):
        return "?"

    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _format_seed_summary(raw_seed_summary: Any) -> str:
    if raw_seed_summary in (None, ""):
        return "<none>"

    if isinstance(raw_seed_summary, str):
        try:
            parsed = json.loads(raw_seed_summary)
        except json.JSONDecodeError:
            return raw_seed_summary
        return json.dumps(parsed, sort_keys=True)

    return json.dumps(raw_seed_summary, sort_keys=True)


def _format_config_json(raw_config_json: Any) -> str:
    if raw_config_json in (None, ""):
        return "<none>"

    if isinstance(raw_config_json, str):
        try:
            parsed = json.loads(raw_config_json)
        except json.JSONDecodeError:
            return raw_config_json
        return json.dumps(parsed, sort_keys=True)

    return json.dumps(raw_config_json, sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_hex__": value.hex()}
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
