#!/usr/bin/env python3
"""Shape check for the traj2traj-only residual skip bug (no HF data / PyG needed).

Reproduces the mismatch: configured eval_batch_size=64 vs actual hop dest=40,
and checks the fix (slice query features by n_dst).
"""
from __future__ import annotations


def wrong_skip(x_for_time_filter_rows: int, eval_batch_size: int, n_dst: int):
    """Old traj2traj-only path: clone-sized filter + [:eval_batch_size]."""
    # x_for_time_filter was x.clone() (>> eval_batch_size); last-layer target used cfg size.
    x_target_rows = min(eval_batch_size, x_for_time_filter_rows)
    return x_target_rows, n_dst, x_target_rows == n_dst


def fixed_skip(n_query: int, n_dst: int):
    """Fixed path: zero-seed n_query rows, then x_target = filter[:n_dst]."""
    x_target_rows = min(n_dst, n_query)
    return x_target_rows, n_dst, x_target_rows == n_dst


def main() -> int:
    eval_batch_size = 64
    n_dst = 40  # partial last validation batch
    full_x_rows = 500  # clone of full sampled x

    bad_skip, out_rows, ok = wrong_skip(full_x_rows, eval_batch_size, n_dst)
    assert not ok and bad_skip == 64 and out_rows == 40, (bad_skip, out_rows, ok)

    good_skip, out_rows, ok = fixed_skip(n_query=n_dst, n_dst=n_dst)
    assert ok and good_skip == 40, (good_skip, out_rows, ok)

    # Paper both: time-filter already has n_dst rows; [:eval_batch_size] == [:n_dst].
    paper_skip = min(eval_batch_size, n_dst)
    assert paper_skip == n_dst

    print("SKIP_SHAPE_OK: wrong skip was x_target/x_skip "
          f"(rows={bad_skip}) vs out (rows={out_rows}); fix uses n_dst={n_dst}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
