"""Build (or check) the downstream submodule's env outside a launch, with its stamp.

``uv run python -m ntt.jobs.downstream_env`` from the host checkout runs the same
check-then-build as a ``job.downstream_env=auto`` launch
(:meth:`~ntt.jobs.downstream.NTTDownstreamJob._ensure_env`), for "build once,
then fan out with ``verify``". Prints the provenance record as JSON.

A separate module (not ``python -m ntt.jobs.downstream``) because ``ntt.jobs``
imports ``downstream`` itself, which ``runpy`` warns about.
"""

from __future__ import annotations

import argparse
import json
import logging

from ntt.jobs.downstream import NTTDownstreamJob


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ntt.jobs.downstream_env",
        description="Build (or check) the downstream submodule's env, with its stamp.",
    )
    parser.add_argument("--root-dir", help="host ntt checkout (default: this one)")
    parser.add_argument(
        "--submodule-dir", default="downstream", help="relative to --root-dir, or absolute"
    )
    parser.add_argument("--verify", action="store_true", help="check only; never build")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    job = NTTDownstreamJob(
        root_dir=args.root_dir,
        submodule_dir=args.submodule_dir,
        downstream_env="verify" if args.verify else "auto",
    )
    host_root = job._host_root()
    record = job._ensure_env((host_root / job.submodule_dir).resolve(), host_root)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
