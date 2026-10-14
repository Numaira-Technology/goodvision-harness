"""Fixed 35B/Alibaba-first parallel scheduling and 48/72/96 comparison.

Use python -m tools.vlm_benchmark.parallel_concurrency --plan-only to inspect.
Same local/cloud lifecycle as qwen35b_alibaba; API calls require --real.
"""

from tools.vlm_benchmark.qwen35b_alibaba import main as fixed_main


def main(argv=None):
    return fixed_main(argv, parallel_sweep=True)


if __name__ == "__main__":
    raise SystemExit(main())
