"""Command-line interface: submit, status, list, resume, cancel, worker.

Two ways to actually execute a job, both documented in the README:

- Foreground/synchronous: `submit --run` (or `resume`) blocks until the job
  reaches COMPLETED or FAILED, in the same terminal. This is what the
  `--offline` demo uses -- no second process required.
- Background/polling: `worker` claims and advances jobs from the queue,
  either draining it once (default) or polling continuously (`--continuous`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .errors import LongRunningAgentError
from .llm import get_client
from .logging_setup import configure_logging
from .models import JobStatus
from .pipeline import DEFAULT_SOURCES
from .runner import execute_job
from .store import JobStore
from .worker import worker_loop

_BAR_WIDTH = 24


def _progress_bar(current: int, total: int) -> str:
    filled = int(_BAR_WIDTH * current / total) if total else 0
    pct = int(100 * current / total) if total else 0
    return f"[{'#' * filled}{'-' * (_BAR_WIDTH - filled)}] {current}/{total} ({pct}%)"


def _load_sources(args: argparse.Namespace) -> list[dict]:
    if args.sources_json:
        return json.loads(args.sources_json)
    if args.sources_file:
        return json.loads(Path(args.sources_file).read_text())
    return DEFAULT_SOURCES


def _print_job_status(store: JobStore, job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None:
        print(f"no job with id {job_id}", file=sys.stderr)
        return
    print(f"job_id={job.id}")
    print(f"status: {job.status.value}")
    print(f"progress: {_progress_bar(job.current_step, job.total_steps)}")
    checkpoints = store.get_checkpoints(job.id)
    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        sources = json.loads(job.input)
        if job.current_step < len(sources):
            next_step = f"process_source:{sources[job.current_step]['name']}"
        elif job.current_step < job.total_steps:
            next_step = "synthesize"
        else:
            next_step = None
        if next_step:
            print(f"current step: {next_step} (next to run)")
    if job.error:
        print(f"error: {job.error}")
    if job.result:
        result = json.loads(job.result)
        synthesis = str(result.get("synthesis", ""))
        trimmed = synthesis if len(synthesis) <= 400 else synthesis[:400] + "..."
        print(f"result: {trimmed}")
    print(f"checkpoints: {len(checkpoints)} step(s) recorded")


def _cmd_submit(args: argparse.Namespace) -> int:
    config = load_config()
    sources = _load_sources(args)
    with JobStore(config.db_path) as store:
        job = store.create_job(sources, max_sources=config.max_sources)
        print(f"job_id={job.id}")
        print(f"status: {job.status.value}")
        print(f"total_steps: {job.total_steps}")
        print(f"sources: {', '.join(s['name'] for s in sources)}")
        if args.run:
            llm_client = get_client(config, force_offline=args.offline)
            finished = execute_job(
                store,
                job.id,
                llm_client,
                max_retries=config.max_retries,
                base_delay=config.retry_base_delay_seconds,
                max_wall_clock_seconds=config.max_wall_clock_seconds,
                simulate_crash_after_step=args.simulate_crash_after_step,
            )
            print("--- run finished ---")
            _print_job_status(store, finished.id)
            return 0 if finished.status is JobStatus.COMPLETED else 1
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    config = load_config()
    with JobStore(config.db_path) as store:
        _print_job_status(store, args.job_id)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    config = load_config()
    with JobStore(config.db_path) as store:
        jobs = store.list_jobs()
        if not jobs:
            print("no jobs yet -- run `submit` first")
            return 0
        for job in jobs:
            print(
                f"{job.id}  {job.status.value:<12} "
                f"{_progress_bar(job.current_step, job.total_steps)}  created={job.created_at}"
            )
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    config = load_config()
    with JobStore(config.db_path) as store:
        llm_client = get_client(config, force_offline=args.offline)
        finished = execute_job(
            store,
            args.job_id,
            llm_client,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay_seconds,
            max_wall_clock_seconds=config.max_wall_clock_seconds,
            simulate_crash_after_step=args.simulate_crash_after_step,
        )
        _print_job_status(store, finished.id)
        return 0 if finished.status is JobStatus.COMPLETED else 1


def _cmd_cancel(args: argparse.Namespace) -> int:
    config = load_config()
    with JobStore(config.db_path) as store:
        job = store.cancel_job(args.job_id)
        print(f"job_id={job.id}")
        print(f"status: {job.status.value}")
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    config = load_config()
    with JobStore(config.db_path) as store:
        llm_client = get_client(config, force_offline=args.offline)
        processed = worker_loop(
            store,
            llm_client,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay_seconds,
            max_wall_clock_seconds=config.max_wall_clock_seconds,
            continuous=args.continuous,
            poll_interval=args.poll_interval,
            max_iterations=args.max_iterations,
        )
        print(f"worker: processed {processed} job(s), exiting")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="long_running_agent")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="create a new job")
    p_submit.add_argument("--sources-file", default=None, help="path to a JSON list of sources")
    p_submit.add_argument("--sources-json", default=None, help="inline JSON list of sources")
    p_submit.add_argument(
        "--run", action="store_true", help="execute the job synchronously right after submitting"
    )
    p_submit.add_argument("--offline", action="store_true", help="force the offline stub model")
    p_submit.add_argument(
        "--simulate-crash-after-step",
        type=int,
        default=None,
        help="hard-exit the process right after committing this step's checkpoint (requires --run)",
    )
    p_submit.set_defaults(func=_cmd_submit)

    p_status = sub.add_parser("status", help="show a job's progress")
    p_status.add_argument("job_id")
    p_status.set_defaults(func=_cmd_status)

    p_list = sub.add_parser("list", help="list every job")
    p_list.set_defaults(func=_cmd_list)

    p_resume = sub.add_parser("resume", help="continue a job from its last checkpoint")
    p_resume.add_argument("job_id")
    p_resume.add_argument("--offline", action="store_true", help="force the offline stub model")
    p_resume.add_argument(
        "--simulate-crash-after-step",
        type=int,
        default=None,
        help="hard-exit the process right after committing this step's checkpoint",
    )
    p_resume.set_defaults(func=_cmd_resume)

    p_cancel = sub.add_parser("cancel", help="cancel a pending or checkpointed job")
    p_cancel.add_argument("job_id")
    p_cancel.set_defaults(func=_cmd_cancel)

    p_worker = sub.add_parser("worker", help="poll for claimable jobs and advance them")
    p_worker.add_argument("--offline", action="store_true", help="force the offline stub model")
    p_worker.add_argument(
        "--continuous", action="store_true", help="keep polling instead of exiting when idle"
    )
    p_worker.add_argument("--poll-interval", type=float, default=1.0)
    p_worker.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="bound on poll iterations in --continuous mode (default 60)",
    )
    p_worker.set_defaults(func=_cmd_worker)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LongRunningAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
