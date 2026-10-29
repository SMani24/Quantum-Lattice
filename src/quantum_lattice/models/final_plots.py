"""Vector final-benchmark figures rendered solely from committed report tables.

Seed standard deviations describe optimizer variation; paired intervals resample
whole Hamiltonians conditional on the frozen models. They are different visual
quantities and are kept in separate figures. This module performs no learning,
sampling, scoring, checkpoint selection or paper-directory writes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quantum-lattice-final-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

SIZES = (3, 4, 8, 12, 16, 20, 24)
ARMS = (("map_mlp", "f"), ("cnn", "f"), ("map_mlp", "g"), ("cnn", "g"))
LABELS = ("MLP / f", "CNN / f", "MLP / g", "CNN / g")
COLORS = ("#68788c", "#0072b2", "#009e73", "#cc79a7")
MARKERS = ("s", "o", "^", "D")
STYLE = {"font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "#182d40", "axes.labelcolor": "#25394b",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "quantum-lattice-final-plots-v1"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(path.read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _inputs(run):
    plan = _read(run / "plan.json")
    config = plan["config"]
    if config["sizes"] != list(SIZES) or len(config["seeds"]) != 5 or len(set(config["seeds"])) != 5:
        raise ValueError("final plots require all seven declared sizes and five paired seeds")
    seeds = config["seeds"]
    with (run / "model-scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells, counts = {}, {}
    for row in rows:
        key = (row["role"],int(row["L"]),row["model"],row["view"],int(row["seed"]))
        if key in cells:
            raise ValueError("duplicate model score cell")
        value = np.asarray([float(row["mse"]),float(row["D5_reference_sign"])])
        if not np.isfinite(value).all() or np.any(value < 0) or value[1] > 2:
            raise ValueError("model scores must be finite nonnegative MSE/unsquared D5")
        count = int(row["hamiltonians"])
        if count < 1 or counts.setdefault(key[:2],count) != count:
            raise ValueError("method/seed Hamiltonian counts differ within a size")
        cells[key] = value
    expected = {(role,L,model,view,seed) for role in ("validation","confirmation")
                for L in SIZES for model,view in ARMS for seed in seeds}
    if set(cells) != expected:
        raise ValueError("score table lacks a complete matched role/size/arm/seed grid")
    contrasts = _read(run / "paired-comparisons.json")["contrasts"]
    effects = {}
    wanted = {"g_minus_f_cnn","cnn_minus_mlp_g"}
    for row in contrasts:
        if row["role"] != "confirmation" or row["metric"] != "mse" or row["comparison"] not in wanted:
            continue
        key = (int(row["L"]),row["comparison"])
        if key in effects:
            raise ValueError("duplicate paired contrast")
        value = np.asarray([row["difference_left_minus_right"],*row["bootstrap_95ci_difference"]],dtype=float)
        expected_primary = key == (4,"g_minus_f_cnn")
        if (value.shape != (3,) or not np.isfinite(value).all() or value[1] > value[2]
                or row.get("statistical_primary") is not expected_primary
                or row["bootstrap"]["confidence_level"] != .95
                or row["hamiltonians"] != counts[("confirmation",key[0])]
                or row["optimizer_seeds"] != len(seeds)):
            raise ValueError("paired contrast interval/scope/count metadata differs")
        effects[key] = value
    if set(effects) != {(L,label) for L in SIZES for label in wanted}:
        raise ValueError("paired contrasts do not cover all declared sizes")
    sampler = config["confirmation"]["sampling"]
    records = (sampler["total_sweeps"]-sampler["burn_in_sweeps"])//sampler["thinning_sweeps"]
    if records < 1:
        raise ValueError("invalid confirmation record budget")
    return plan,cells,counts,effects,records


def _bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width,height = fig.canvas.get_width_height()
    # Locators can retain visible Text objects for ticks outside the view.
    # Axis.draw uses _update_ticks() to omit those ticks; inspect the same set
    # so an undrawn label cannot falsely fail the canvas check.
    undrawn_tick_labels = set()
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            drawn_ticks = set(axis._update_ticks())
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if tick not in drawn_ticks:
                    undrawn_tick_labels.update((tick.label1, tick.label2))
    missing = []
    for text in fig.findobj(match=Text):
        if text not in undrawn_tick_labels and text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
                missing.append(text.get_text())
    if missing:
        raise ValueError(f"figure text extends outside the canvas: {missing}")


def _save(fig,directory,stem,pdf):
    _bounds(fig)
    paths=[]
    for extension in ("pdf","svg","png"):
        path=directory/f"{stem}.{extension}"
        metadata = {"CreationDate":None,"ModDate":None} if extension=="pdf" else {"Date":None} if extension=="svg" else None
        fig.savefig(path,dpi=300,metadata=metadata,facecolor="white")
        paths.append(path)
    pdf.savefig(fig)
    plt.close(fig)
    return paths


def _count_label(counts,role):
    values=[counts[(role,L)] for L in SIZES]
    return f"{values[0]:,}" if len(set(values))==1 else "/".join(f"{n:,}" for n in values)


def render_final_plots(run):
    """Export two matched multi-size figures and a vector PDF gallery.

    Requires completed ``model-scores.csv``, ``paired-comparisons.json`` and the
    frozen ``plan.json``. Optional top-level plan ``synthetic_fixture=true`` is
    only for explicitly watermarked renderer tests. Original tables are never
    changed. Existing outputs are reused only when all input/output hashes match.
    Returns a dict mapping every exported filename to its absolute path.
    """
    run=Path(run).resolve()
    input_names=("plan.json","model-scores.csv","paired-comparisons.json")
    request={"schema_version":1,"source_sha256":_sha(Path(__file__)),
             "input_sha256":{name:_sha(run/name) for name in input_names},
             "numpy_version":np.__version__,"matplotlib_version":matplotlib.__version__}
    directory=run/"figures"
    manifest_path=directory/"plot-provenance.json"
    if manifest_path.exists():
        manifest=_read(manifest_path)
        if manifest.get("request")!=request:
            raise ValueError("plot sources/inputs changed; use a new run output")
        if any(_sha(directory/name)!=digest for name,digest in manifest["output_sha256"].items()):
            raise ValueError("plot output checksum mismatch")
        return {name:str(directory/name) for name in [*manifest["output_sha256"],manifest_path.name]}
    plan,cells,counts,effects,records=_inputs(run)
    seeds=plan["config"]["seeds"]
    fixture=plan.get("synthetic_fixture") is True
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    with plt.rc_context(STYLE), PdfPages(directory/"final-figure-gallery.pdf",metadata={"CreationDate":None}) as pdf:
        fig=plt.figure(figsize=(7.1,5.8),layout="constrained")
        grid=fig.add_gridspec(3,2,height_ratios=[.17,1,1])
        legend=fig.add_subplot(grid[0,:]);legend.axis("off")
        handles=[Line2D([],[],color=c,marker=m,lw=1.3,markersize=4,label=label)
                 for c,m,label in zip(COLORS,MARKERS,LABELS)]
        legend.legend(handles=handles,loc="center",ncol=4,frameon=False,fontsize=8)
        if fixture:fig.suptitle("SYNTHETIC TEST FIXTURE — no research results",fontsize=9)
        x=np.arange(len(SIZES),dtype=float)
        offsets=np.linspace(-.18,.18,4)
        for col,role in enumerate(("validation","confirmation")):
            for row,(index,ylabel) in enumerate(((0,"Field-magnitude MSE"),(1,"Reference-sign D5 (unsquared)"))):
                ax=fig.add_subplot(grid[row+1,col])
                minimum=0.
                for arm,offset,color,marker in zip(ARMS,offsets,COLORS,MARKERS):
                    values=np.asarray([[cells[(role,L,*arm,seed)][index] for seed in seeds] for L in SIZES])
                    mean,sd=values.mean(axis=1),values.std(axis=1,ddof=1)
                    minimum=min(minimum,float(np.min(mean-sd)))
                    ax.errorbar(x+offset,mean,yerr=sd,color=color,marker=marker,lw=1.2,ms=3.5,capsize=2,zorder=3)
                    for i,position in enumerate(x+offset):
                        ax.scatter(position+np.linspace(-.025,.025,len(seeds)),values[i],s=6,color=color,alpha=.5,zorder=4)
                ax.set_xticks(x,[str(L) for L in SIZES])
                ax.set_ylim(bottom=minimum*1.08 if minimum<0 else 0)
                ax.set_xlabel("Side length L (categorical spacing)")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y",alpha=.16)
                title="Development validation" if role=="validation" else "Fresh confirmation"
                ax.set_title(f"({'abcd'[2*row+col]}) {title}",loc="left",pad=9)
        fig.supxlabel(f"Known-field simulation features · confirmation M={records:,} retained records/H\n"
                      f"N_H per size: validation {_count_label(counts,'validation')}; confirmation {_count_label(counts,'confirmation')}\n"
                      "Dots: all 5 seeds · lines ± seed SD, not a confidence interval\n"
                      "D5 grants reference signs, full two-channel, unsquared · lower is better",fontsize=8)
        paths+=_save(fig,directory,"observable-performance",pdf)

        fig,axes=plt.subplots(1,2,figsize=(7.1,3.65),layout="constrained")
        if fixture:fig.suptitle("SYNTHETIC TEST FIXTURE — no research results",fontsize=9)
        specifications=(("g_minus_f_cnn","(a) Estimator: CNN g − f"),
                        ("cnn_minus_mlp_g","(b) Architecture: CNN − MLP (g)"))
        for ax,(label,title) in zip(axes,specifications):
            values=np.asarray([effects[(L,label)] for L in SIZES])
            ax.axhline(0,color="#68788c",lw=1,ls="--",zorder=1)
            if label=="g_minus_f_cnn":ax.axvspan(.7,1.3,color="#fff0c5",zorder=0)
            ax.vlines(x,values[:,1],values[:,2],color=COLORS[3],lw=1.4,zorder=3)
            ax.hlines(values[:,1],x-.065,x+.065,color=COLORS[3],lw=1)
            ax.hlines(values[:,2],x-.065,x+.065,color=COLORS[3],lw=1)
            ax.scatter(x,values[:,0],s=20,color=COLORS[3],marker="o",zorder=4)
            if label=="g_minus_f_cnn":
                i=SIZES.index(4)
                ax.scatter([x[i]],[values[i,0]],s=52,facecolors="none",edgecolors="#182d40",lw=1.2,zorder=5)
            ax.set_xticks(x,[str(L) for L in SIZES])
            ax.set_xlabel("Side length L (categorical spacing)")
            ax.set_ylabel("Paired magnitude-MSE difference")
            ax.set_title(title,loc="left",pad=10)
            ax.grid(axis="y",alpha=.16)
            ax.margins(y=.18)
        fig.supxlabel("Fresh confirmation · seed-averaged losses within each H · negative favors the first method\n"
                      "Bars: paired whole-H 95% intervals conditional on the frozen models/training bank\n"
                      f"Known-field features · M={records:,} · circled L4 in (a) is primary; all other contrasts secondary",fontsize=8)
        paths+=_save(fig,directory,"paired-effects",pdf)
    paths.append(directory/"final-figure-gallery.pdf")
    manifest={"schema_version":1,"request":request,"synthetic_fixture":fixture,
              "source_table_scope":"all own-view validation/confirmation cells; seven sizes, four arms, five seeds",
              "seed_uncertainty":"performance bars are sample SD across optimizer seeds, not confidence intervals",
              "paired_uncertainty":"whole-H percentile95% conditional on fixed models/training bank; no seed/H pseudo-replication",
              "text_bounds_checked":True,"figure_width_inches":7.1,
              "output_sha256":{p.name:_sha(p) for p in paths}}
    temporary=manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    temporary.replace(manifest_path)
    return {p.name:str(p) for p in [*paths,manifest_path]}
