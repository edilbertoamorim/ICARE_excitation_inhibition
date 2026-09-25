"""Run everything for the variable set chosen in config.py: fit, tables, figure.

    python3 run_all.py            # whatever config.VARS says
    python3 run_all.py X          # override it for one run
    python3 run_all.py X XY       # both, one after the other

A full run is ~25 min for X and ~35 min for XY; `python3 smoke_test.py` first.
"""

import sys
import time

import config as cfg
import figure
import gbtm


def main(tags: list[str]) -> None:
    for tag in tags or [cfg.VARS]:
        tag = cfg.check(tag)
        print(f"\n=== GBTM on {'+'.join(cfg.VAR_SETS[tag])} "
              f"({cfg.N_STARTS} starts, {cfg.N_BOOT} bootstrap refits) ===")
        t0 = time.time()
        gbtm.run(tag)
        print(f"\n  model and tables: {(time.time() - t0) / 60:.1f} min")
        figure.draw(tag)
    print("\nDone. Tables are in results/, the figure in figures/.")


if __name__ == "__main__":
    main([a.upper() for a in sys.argv[1:]])
