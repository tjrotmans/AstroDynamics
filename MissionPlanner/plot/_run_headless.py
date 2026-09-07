"""Headless runner for plot_mission_sim.py — replaces TkAgg with Agg."""
import sys, os
from pathlib import Path

script = Path(__file__).parent / "plot_mission_sim.py"
src = script.read_text(encoding="utf-8")
src = src.replace("matplotlib.use(\"TkAgg\")", "matplotlib.use(\"Agg\")")
src = src.replace("plt.show()", "# plt.show()")

sys.argv = [str(script)]
exec(compile(src, str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
