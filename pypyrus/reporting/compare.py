"""
Cross-run comparison helpers.

Comparisons are per-role: train batches in run A are compared against
train batches in run B, val against val, etc.  Within each role the
batches are ordered by global_step (per-loader cursor).
"""

from __future__ import annotations

from typing import Any

from pypyrus.reporting.queries import decode_sample_ids_blob, get_batches_for_run
from pypyrus.storage.store import Store


def _get_roles(store: Store, run_id: str) -> set[str]:
    rows = store.get_events(run_id, event_type="dataset_registered")
    return {r["role"] for r in rows if r.get("role")}


def _get_datasets_by_role(store: Store, run_id: str) -> dict[str, list[dict[str, Any]]]:
    rows = store.get_events(run_id, event_type="dataset_registered")
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        role = row.get("role")
        if not role:
            continue
        by_role.setdefault(role, []).append(
            {
                "dataset_id": row.get("dataset_id"),
                "name": row.get("name"),
                "uri": row.get("uri"),
                "fingerprint": row.get("fingerprint"),
                "fingerprint_method": row.get("fingerprint_method"),
            }
        )

    for role, datasets in by_role.items():
        datasets.sort(
            key=lambda d: (
                str(d.get("dataset_id") or ""),
                str(d.get("fingerprint") or ""),
            )
        )

    return by_role


def _compare_role_dataset_identity(
    role: str,
    datasets_a: list[dict[str, Any]],
    datasets_b: list[dict[str, Any]],
) -> dict[str, Any]:
    if datasets_a == datasets_b:
        return {
            "role": role,
            "matches": True,
            "run_a": datasets_a,
            "run_b": datasets_b,
            "reason": None,
        }

    reason = "dataset_count_mismatch" if len(datasets_a) != len(datasets_b) else "dataset_fingerprint_mismatch"
    return {
        "role": role,
        "matches": False,
        "run_a": datasets_a,
        "run_b": datasets_b,
        "reason": reason,
    }


def compare_runs(
    store: Store,
    run_id_a: str,
    run_id_b: str,
) -> dict[str, Any]:
    """
    Compare two runs per role.

    For each role present in either run, compare the per-loader batch
    streams (ordered by global_step).  Also reports global_sequence so
    the position of each role's batches in the full run timeline is
    visible.

    Returns a dict with:
        run_id_a, run_id_b
        roles:  dict[role -> per-role comparison result]
        fully_matches: True if every role fully matches
    """
    roles_a = _get_roles(store, run_id_a)
    roles_b = _get_roles(store, run_id_b)
    all_roles = sorted(roles_a | roles_b)
    datasets_by_role_a = _get_datasets_by_role(store, run_id_a)
    datasets_by_role_b = _get_datasets_by_role(store, run_id_b)

    role_results: dict[str, Any] = {}
    for role in all_roles:
        dataset_identity = _compare_role_dataset_identity(
            role,
            datasets_by_role_a.get(role, []),
            datasets_by_role_b.get(role, []),
        )

        batches_a = get_batches_for_run(store, run_id_a, include_sample_ids=True, role=role)
        batches_b = get_batches_for_run(store, run_id_b, include_sample_ids=True, role=role)

        # Sort each by global_step (per-loader cursor) for comparison
        batches_a.sort(key=lambda r: r["global_step"])
        batches_b.sort(key=lambda r: r["global_step"])

        role_result = _compare_role_batches(role, batches_a, batches_b)
        role_result["dataset_identity"] = dataset_identity
        role_result["dataset_identity_matches"] = dataset_identity["matches"]
        role_result["fully_matches"] = role_result["fully_matches"] and dataset_identity["matches"]

        role_results[role] = role_result

    fully_matches = all(r["fully_matches"] for r in role_results.values())

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "roles": role_results,
        "fully_matches": fully_matches,
    }


def _compare_role_batches(
    role: str,
    batches_a: list[dict[str, Any]],
    batches_b: list[dict[str, Any]],
) -> dict[str, Any]:
    len_a = len(batches_a)
    len_b = len(batches_b)
    min_len = min(len_a, len_b)

    matching_steps = 0
    first_divergence_step: int | None = None
    divergence_a: dict[str, Any] | None = None
    divergence_b: dict[str, Any] | None = None

    for i in range(min_len):
        if batches_a[i]["batch_fingerprint"] == batches_b[i]["batch_fingerprint"]:
            matching_steps += 1
        else:
            first_divergence_step = i
            divergence_a = _batch_view(batches_a[i])
            divergence_b = _batch_view(batches_b[i])
            break

    if first_divergence_step is None and len_a != len_b:
        first_divergence_step = min_len
        divergence_a = _batch_view(batches_a[min_len]) if len_a > min_len else None
        divergence_b = _batch_view(batches_b[min_len]) if len_b > min_len else None

    fully_matches = (len_a == len_b) and (matching_steps == min_len)
    match_rate = matching_steps / max(len_a, len_b, 1)

    return {
        "role": role,
        "num_batches_a": len_a,
        "num_batches_b": len_b,
        "matching_steps": matching_steps,
        "match_rate": match_rate,
        "first_divergence_step": first_divergence_step,
        "fully_matches": fully_matches,
        "divergence": {
            "run_a": divergence_a,
            "run_b": divergence_b,
        } if first_divergence_step is not None else None,
    }


def _batch_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "loader_id": row.get("loader_id"),
        "role": row.get("role"),
        "global_step": row["global_step"],
        "global_sequence": row.get("global_sequence"),
        "batch_size": row["batch_size"],
        "batch_fingerprint": row["batch_fingerprint"],
        "sample_ids": row.get("sample_ids"),
    }


def format_run_comparison(result: dict[str, Any]) -> str:
    """Pretty-print a compare_runs result for terminal viewing."""
    lines: list[str] = []
    lines.append("Run comparison")
    lines.append("-" * 60)
    lines.append(f"Run A: {result['run_id_a']}")
    lines.append(f"Run B: {result['run_id_b']}")
    lines.append(f"Fully matches: {result['fully_matches']}")
    lines.append("")

    for role, r in result["roles"].items():
        lines.append(f"Role: {role}")
        lines.append(f"  Batches A / B:   {r['num_batches_a']} / {r['num_batches_b']}")
        lines.append(f"  Dataset match:   {r['dataset_identity_matches']}")
        dataset_identity = r["dataset_identity"]
        if not dataset_identity["matches"]:
            lines.append(f"  Dataset reason:  {dataset_identity['reason']}")
        lines.append(f"  Matching steps:  {r['matching_steps']}")
        lines.append(f"  Match rate:      {r['match_rate']:.2%}")
        lines.append(f"  Fully matches:   {r['fully_matches']}")

        if r["first_divergence_step"] is not None:
            lines.append(f"  First divergence: step {r['first_divergence_step']}")
            div = r["divergence"]
            for label, batch in (("Run A", div["run_a"]), ("Run B", div["run_b"])):
                lines.extend(_format_one_batch(label, batch))
        else:
            lines.append("  First divergence: None")

        lines.append("")

    return "\n".join(lines)


def _format_one_batch(label: str, batch: dict[str, Any] | None) -> list[str]:
    if batch is None:
        return [f"  {label}: <missing batch>"]
    sample_ids = batch.get("sample_ids")
    ids_str = ", ".join(map(str, sample_ids)) if sample_ids else "<unavailable>"
    return [
        f"  {label}:",
        f"    loader_id:       {batch.get('loader_id')}",
        f"    role:            {batch.get('role')}",
        f"    global_sequence: {batch.get('global_sequence')}",
        f"    global_step:     {batch['global_step']}",
        f"    batch_size:      {batch['batch_size']}",
        f"    fingerprint:     {batch['batch_fingerprint']}",
        f"    sample_ids:      [{ids_str}]",
    ]
