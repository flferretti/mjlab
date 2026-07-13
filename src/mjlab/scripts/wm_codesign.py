"""Run the NSGA-II co-design loop with the world-model surrogate backend.

Thin wrapper around ``scripts/codesign_ga.py`` that injects ``--backend wm``;
all other flags are forwarded unchanged, e.g.::

  uv run wm-codesign --wm-checkpoint logs/world_model/qdd_v1/.../wm_final.pt \\
    --policy ./model_30000.pt --pop-size 32 --generations 25 --multiobjective \\
    --cost-model powerlaw

The GA script lives in the repository's ``scripts/`` directory (it is not part
of the installed package), so this command must run from a checkout.
"""

import runpy
import sys
from pathlib import Path


def _find_ga_script() -> Path:
  candidates = [Path.cwd(), *Path.cwd().parents]
  # Also try relative to the installed package (editable installs).
  candidates.append(Path(__file__).resolve().parents[3])
  for root in candidates:
    script = root / "scripts" / "codesign_ga.py"
    if script.exists():
      return script
  raise SystemExit(
    "Could not locate scripts/codesign_ga.py; run wm-codesign from an mjlab "
    "repository checkout."
  )


def main() -> None:
  script = _find_ga_script()
  argv = sys.argv[1:]
  if "--backend" not in argv:
    argv = ["--backend", "wm", *argv]
  sys.path.insert(0, str(script.parent))
  sys.argv = [str(script), *argv]
  runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
  main()
