"""Export the notebooks to runnable .py scripts (the notebooks stay the source of truth).

- a cell's last bare expression (a table, a value) is wrapped in print(), as a notebook would display it
- plt.show() saves each figure to figures/<script>_NN.png instead of opening a window
Usage: .venv/bin/python scripts/export_scripts.py      (notebooks/*.ipynb → scripts/*.py)
"""
import ast
import json
from pathlib import Path

HEADER = '''# Auto-generated from {nb} by export_scripts.py — edit the notebook, then re-export.
# Run:  .venv/bin/python scripts/{py}
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[1])   # always run from the project root (data/, src/)
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
_FIG_DIR = _Path("figures"); _FIG_DIR.mkdir(exist_ok=True)
_fig_n = 0


def _save_show(*args, **kwargs):
    global _fig_n
    for num in _plt.get_fignums():
        _fig_n += 1
        _plt.figure(num).savefig(_FIG_DIR / ("{stem}_%02d.png" % _fig_n), dpi=120, bbox_inches="tight")
    _plt.close("all")


_plt.show = _save_show
'''


def convert_cell(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return src
    last = tree.body[-1]
    call = last.value
    if isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", "")) in {"print", "show", "plot", "heatmap", "mkdir", "append", "update"}:
        return src
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in {"barh", "bar", "hist", "kde"}:
        return src
    lines = src.split("\n")
    s, e = last.lineno - 1, last.end_lineno - 1
    head, tail = lines[s][: last.col_offset], lines[e][last.end_col_offset:]
    body = lines[s:e + 1]
    body[-1] = body[-1][: last.end_col_offset]
    body[0] = body[0][last.col_offset:]
    return "\n".join(lines[:s] + [head + "print(" + "\n".join(body) + ")" + tail] + lines[e + 1:])


ROOT = Path(__file__).resolve().parents[1]
for nb_path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
    nb = json.loads(nb_path.read_text())
    py_path = ROOT / "scripts" / (nb_path.stem + ".py")
    parts = [HEADER.format(nb=nb_path.name, py=py_path.name, stem=nb_path.stem)]
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append("\n".join("# " + l if l else "#" for l in src.split("\n")))
        elif c["cell_type"] == "code" and src.strip():
            parts.append(convert_cell(src.replace("# !pip", "# pip")))
    py_path.write_text("\n\n\n".join(parts) + "\n")
    compile(py_path.read_text(), str(py_path), "exec")
    print("wrote", py_path)
