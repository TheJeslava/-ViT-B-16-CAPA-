import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from train import build_ablation_runs, load_config, run_experiment  # noqa: E402


DEFAULT_DATASETS = [
    "cifar100_full",
    "vtab1k_cifar100",
    "cub_200_2011",
    "oxford_flowers_102",
    "snorb_azim",
    "snorb_elev",
]

SUMMARY_FIELDS = [
    "status",
    "group",
    "dataset",
    "method",
    "run_name",
    "seed",
    "best_epoch",
    "best_val_top1",
    "test_top1_at_best_val",
    "trainable_params_m",
    "trainable_params_percent_of_86m",
    "inference_flops_g",
    "peak_train_memory_gb",
    "mean_epoch_time_s",
    "router_mean_entropy",
    "router_effective_experts",
    "expert_load_min",
    "expert_load_max_min_ratio",
    "summary_path",
    "completed_at",
]


@dataclass
class Job:
    group: str
    dataset: str
    method: str
    run_name: str
    overrides: Optional[Dict] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run all Para-CVPT experiments sequentially.")
    parser.add_argument("--config", default="configs/para_cvpt_single_seed.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to run. Default: all 4 main datasets.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run. Default: config methods.main_table.")
    parser.add_argument("--include-ablations", action="store_true", help="Also run configured ablations after the main table.")
    parser.add_argument(
        "--ablation-groups",
        nargs="+",
        default=["core_structure", "collapse_guard", "capacity", "scope"],
        choices=["core_structure", "collapse_guard", "capacity", "scope"],
    )
    parser.add_argument("--ablation-dataset", default=None, help="Dataset for ablations. Defaults to each group dataset_first.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Run even if summary.json already exists.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed job.")
    parser.add_argument("--dry-run", action="store_true", help="Print the schedule without running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root or config.get("outputs", {}).get("root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else config.get("training", {}).get("seed", 0))
    jobs = build_jobs(config, args)
    print_schedule(jobs, seed)
    if args.dry_run:
        return

    train_args = SimpleNamespace(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        output_root=str(output_root),
        device=args.device,
        download=args.download,
        no_pretrained=args.no_pretrained,
        limit_train_batches=args.limit_train_batches,
        limit_eval_batches=args.limit_eval_batches,
    )

    completed: List[Dict] = load_existing_summaries(output_root, jobs, seed)
    write_cumulative_table(tables_dir / "full_results.csv", completed)
    manifest_path = tables_dir / "full_run_manifest.jsonl"

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        summary_path = get_summary_path(output_root, job, seed)
        print(f"\n[{idx}/{total}] {job.group} | {job.dataset} | {job.run_name} ({job.method})", flush=True)

        if summary_path.exists() and not args.force:
            print(f"Skip existing: {summary_path}", flush=True)
            summary = load_summary(summary_path)
            record = make_record(job, summary, summary_path, status="skipped_existing")
            completed = upsert_record(completed, record)
            write_cumulative_table(tables_dir / "full_results.csv", completed)
            append_manifest(manifest_path, record)
            continue

        start = time.time()
        try:
            summary = run_experiment(
                base_config=config,
                method=job.method,
                dataset_key=job.dataset,
                run_name=job.run_name,
                args=train_args,
                config_overrides=job.overrides,
            )
            record = make_record(job, summary, summary_path, status="completed")
            record["wall_time_s"] = round(time.time() - start, 3)
            completed = upsert_record(completed, record)
            write_cumulative_table(tables_dir / "full_results.csv", completed)
            append_manifest(manifest_path, record)
            print(f"Archived: {summary_path}", flush=True)
            print(f"Updated: {tables_dir / 'full_results.csv'}", flush=True)
        except Exception as exc:
            record = {
                "status": "failed",
                "group": job.group,
                "dataset": job.dataset,
                "method": job.method,
                "run_name": job.run_name,
                "seed": seed,
                "summary_path": str(summary_path),
                "completed_at": timestamp(),
                "error": repr(exc),
            }
            append_manifest(manifest_path, record)
            write_failure(tables_dir / "full_run_failures.jsonl", record)
            print(f"FAILED: {repr(exc)}", flush=True)
            if not args.keep_going:
                raise


def build_jobs(config: Dict, args) -> List[Job]:
    datasets = args.datasets or DEFAULT_DATASETS
    methods = args.methods or config.get("methods", {}).get("main_table", [])
    jobs = [
        Job(group="main", dataset=dataset, method=method, run_name=method)
        for dataset in datasets
        for method in methods
    ]

    if args.include_ablations:
        dummy_args = SimpleNamespace()
        for group in args.ablation_groups:
            group_cfg = config.get("ablations", {}).get(group, {})
            dataset = args.ablation_dataset or group_cfg.get("dataset_first", config.get("experiment", {}).get("dataset"))
            for run_name, method, overrides in build_ablation_runs(config, group, dummy_args):
                clean_name = run_name
                if clean_name.startswith(f"{group}_"):
                    clean_name = clean_name[len(group) + 1 :]
                jobs.append(
                    Job(
                        group=group,
                        dataset=dataset,
                        method=method,
                        run_name=f"ablation_{group}_{clean_name}",
                        overrides=overrides,
                    )
                )
    return jobs


def print_schedule(jobs: List[Job], seed: int) -> None:
    print(f"Sequential schedule: {len(jobs)} jobs, single seed={seed}", flush=True)
    for idx, job in enumerate(jobs, start=1):
        print(f"{idx:03d}. {job.group:15s} {job.dataset:20s} {job.run_name:28s} method={job.method}", flush=True)


def get_summary_path(output_root: Path, job: Job, seed: int) -> Path:
    return output_root / job.dataset / job.run_name / f"seed_{seed}" / "summary.json"


def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_existing_summaries(output_root: Path, jobs: Iterable[Job], seed: int) -> List[Dict]:
    records: List[Dict] = []
    for job in jobs:
        path = get_summary_path(output_root, job, seed)
        if path.exists():
            records.append(make_record(job, load_summary(path), path, status="existing_at_start"))
    return records


def make_record(job: Job, summary: Dict, summary_path: Path, status: str) -> Dict:
    record = {field: "" for field in SUMMARY_FIELDS}
    record.update(summary)
    record.update(
        {
            "status": status,
            "group": job.group,
            "dataset": job.dataset,
            "method": job.method,
            "run_name": job.run_name,
            "summary_path": str(summary_path),
            "completed_at": timestamp(),
        }
    )
    return record


def upsert_record(records: List[Dict], new_record: Dict) -> List[Dict]:
    key = (new_record.get("group"), new_record.get("dataset"), new_record.get("run_name"), new_record.get("seed"))
    out = []
    replaced = False
    for record in records:
        record_key = (record.get("group"), record.get("dataset"), record.get("run_name"), record.get("seed"))
        if record_key == key:
            out.append(new_record)
            replaced = True
        else:
            out.append(record)
    if not replaced:
        out.append(new_record)
    return out


def write_cumulative_table(path: Path, records: List[Dict]) -> None:
    extra_fields = sorted({key for record in records for key in record.keys()} - set(SUMMARY_FIELDS))
    fields = SUMMARY_FIELDS + extra_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def append_manifest(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_failure(path: Path, record: Dict) -> None:
    append_manifest(path, record)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
