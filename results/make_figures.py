"""Generate the paper's figures from the canonical results (results_data.py).

Writes 300-dpi PNGs to figures/. Run: python results/make_figures.py
--geometry also prints the Fig. 1 map layout (JSON) used by the paper's TikZ wrapper.

The paper typesets Figs. 1 and 3 to 5 natively in TikZ from the same numbers, so
these PNGs are raster versions of them. Only fig_studyarea_both_bare() feeds the
paper directly (the Fig. 1 reach rasters and their geometry).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_data as R

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    # Nimbus Roman is the URW Times clone pdflatex substitutes for the IEEEtran
    # body font, so raster figures match the paper's typography.
    "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "Times New Roman"],
    "mathtext.fontset": "stix",
})
GRAY, RED, BLUE = "#9aa0a6", "#b3261e", "#1f5fa8"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p))


def fig_cofluctuation():
    """Fig. 3: co-fluctuation advantage (flow minus distance-matched control) with
    95% CI per index and deseasoning, Germany vs Romania."""
    from matplotlib.lines import Line2D
    ORANGE = "#d95f0e"  # Romania (Germany = BLUE)
    sig_label = {"Turbidity": "turbidity", "NDCI": "NDCI"}
    des_label = {"common-mode": "common-mode", "seasonal": "seasonal"}
    rows = [("NDCI", "common-mode"), ("NDCI", "seasonal"),
            ("Turbidity", "common-mode"), ("Turbidity", "seasonal")]
    RO = R.ROMANIA_REPL["cofluctuation"]
    y = np.arange(len(rows))[::-1]; off = 0.17
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    fig.subplots_adjust(left=0.30, right=0.965, top=0.94, bottom=0.26)
    # dashed zero reference just inside the left edge
    ax.axvline(0, color="#888", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for yi, key in zip(y, rows):
        d, lo, hi = R.COFLUCTUATION[key][2:5]
        dr, lor, hir = RO[key][2:5]
        ax.plot([lo, hi], [yi + off] * 2, color=BLUE, lw=2.6, solid_capstyle="round", zorder=2)
        ax.plot(d, yi + off, "o", color=BLUE, mec="white", mew=0.6, ms=6, zorder=3)
        ax.plot([lor, hir], [yi - off] * 2, color=ORANGE, lw=2.6, solid_capstyle="round", zorder=2)
        ax.plot(dr, yi - off, "s", color=ORANGE, mec="white", mew=0.6, ms=5.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{sig_label[s]}\n{des_label[de]}" for s, de in rows], fontsize=7.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.012, 0.185)
    ax.set_xlabel("co-fluctuation advantage\n(flow $-$ distance-matched), 95% CI", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    handles = [Line2D([0], [0], color=BLUE, marker="o", lw=2.4, ms=6, mec="white",
                      mew=0.6, label="Germany"),
               Line2D([0], [0], color=ORANGE, marker="s", lw=2.4, ms=5.5, mec="white",
                      mew=0.6, label="Romania")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.8, frameon=False,
              handletextpad=0.4, labelspacing=0.3)
    _save(fig, "fig_cofluctuation.png")


def fig_resolution():
    """Fig. 5: (a) graph minus proximity effect and (b) share of empty cells vs
    temporal bin width, read from results/national/resolution_sweep.json."""
    import json
    _sweep_path = os.path.join(os.path.dirname(__file__), "..", "results",
                               "national", "resolution_sweep.json")
    with open(_sweep_path, encoding="utf-8") as fh:
        sweep = json.load(fh)
    W = [1, 3, 7, 15, 28]
    xs = list(range(len(W)))

    def series(path):
        out = []
        for w in W:
            node = sweep[str(w)]
            for k in path.split("/"):
                node = node[k]
            out.append(node)
        return out

    from matplotlib.lines import Line2D
    # Co-fluctuation: green, solid, shaded CI band. Synthetic event tests: blue/grey,
    # dashed, no band. Filled marker = 95% CI excludes zero.
    CHL, TURB, REC, BRK = "#1b6b2e", "#66bd63", BLUE, "#8a8f94"
    series_defs = [  # (label, mean, lo, hi, colour, linestyle, shaded_band)
        ("NDCI ($\\Delta$corr)", "cofluctuation_ndci/diff",
         "cofluctuation_ndci/lo", "cofluctuation_ndci/hi", CHL, "-", True),
        ("Turbidity ($\\Delta$corr)", "cofluctuation_turb/diff",
         "cofluctuation_turb/lo", "cofluctuation_turb/hi", TURB, "-", True),
        ("Event recovery ($\\Delta$F1)", "recovery/delta/mean",
         "recovery/delta/lo", "recovery/delta/hi", REC, (0, (5, 2)), False),
        ("Neighbour-break ($\\Delta$ROC-AUC)", "break/vs_euc/mean",
         "break/vs_euc/lo", "break/vs_euc/hi", BRK, (0, (5, 2)), False),
    ]
    # two panels side by side within a single column: (a) effect, (b) sparsity
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(3.5, 2.02),
                                  gridspec_kw={"width_ratios": [1.5, 1.05]})
    ax.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    ax.axhline(0, color="#888", lw=0.8, ls=(0, (4, 3)), zorder=1)
    handles = []
    for lab, mp, lp, hp, col, ls, band in series_defs:
        m, lo, hi = series(mp), series(lp), series(hp)
        if band:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, m, ls=ls, color=col, lw=2.0 if band else 1.6, zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        for k in xs:
            sig = lo[k] > 0
            ax.plot(k, m[k], "o", mfc=(col if sig else "white"), mec=col, mew=1.1,
                    ms=5.0, zorder=4)
        handles.append(Line2D([0], [0], color=col, ls=ls, lw=2.0, marker="o",
                              mfc=col, mec=col, ms=5, label=lab))
    ax.set_ylabel("graph $-$ proximity effect", fontsize=8.4)
    ax.set_xticks(xs); ax.set_xticklabels([str(w) for w in W])
    ax.set_xlim(-0.35, len(W) - 0.65)
    ax.tick_params(labelsize=7.8)
    ax.text(0.035, 0.965, "(a)", transform=ax.transAxes, fontsize=9.5, va="top",
            fontweight="bold")

    empty = [100.0 * v for v in series("coverage/nan_rate")]
    axb.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    axb.fill_between(xs, 0, empty, color=GRAY, alpha=0.28, lw=0, zorder=2)
    axb.plot(xs, empty, "-", color="#555", lw=1.7, marker="s", ms=4.3, zorder=3)
    axb.annotate(f"{empty[0]:.0f}%", (xs[0], empty[0]), textcoords="offset points",
                 xytext=(7, 1), fontsize=8.0, color="#444")
    axb.annotate(f"{empty[-1]:.0f}%", (xs[-1], empty[-1]), textcoords="offset points",
                 xytext=(-2, 11), fontsize=8.0, color="#444", ha="right")
    axb.set_ylim(0, 100)
    axb.set_ylabel("no-pass cells (%)", fontsize=8.4)
    axb.set_xticks(xs); axb.set_xticklabels([str(w) for w in W])
    axb.set_xlim(-0.35, len(W) - 0.65)
    axb.tick_params(labelsize=7.8)
    axb.text(0.94, 0.965, "(b)", transform=axb.transAxes, fontsize=9.5, va="top",
             ha="right", fontweight="bold")

    fig.subplots_adjust(left=0.135, right=0.975, top=0.95, bottom=0.32, wspace=0.5)
    # one shared x-axis label centred over both panels
    fig.text(0.555, 0.185, "temporal bin width (days)", ha="center", fontsize=8.4)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.0, frameon=False,
               handlelength=1.3, columnspacing=0.8, labelspacing=0.35,
               handletextpad=0.4, bbox_to_anchor=(0.555, 0.0))
    _save(fig, "fig_resolution.png")


def fig_studyarea_both():
    """Fig. 1: Germany and Romania stacked in one column at one true scale.

    Both panels use x = lon * cos(mean latitude), y = lat at the same inches per
    unit, so one projected unit is ~111 km in both. Germany shows its 209 in-situ
    labelled reaches. One Strahler-order colourbar spans both panels."""
    import csv as _csv
    import json
    import math
    DATA = os.path.join(os.path.dirname(__file__), "..", "data")

    def _load(country):
        with open(os.path.join(DATA, country, f"segment_graph_{country}.json"),
                  encoding="utf-8") as fh:
            g = json.load(fh)
        lon = np.array([v["lon"] for v in g.values()], float)
        lat = np.array([v["lat"] for v in g.values()], float)
        st = np.array([(v.get("strahler") or 1) for v in g.values()], float)
        return g, lon, lat, st

    gde, lon_de, lat_de, st_de = _load("de")
    gro, lon_ro, lat_ro, st_ro = _load("ro")
    # Germany's 209 monitored surface-water reaches (distinct labelled reaches)
    with open(os.path.join(DATA, "de", "strong_labels_national_de.csv"),
              encoding="utf-8") as fh:
        labeled = sorted(set(r["object_id"] for r in _csv.DictReader(fh)))
    slon = np.array([gde[o]["lon"] for o in labeled if o in gde], float)
    slat = np.array([gde[o]["lat"] for o in labeled if o in gde], float)

    vmin, vmax = 0.0, float(max(st_de.max(), st_ro.max()))

    # equal-scale projection: one projected unit is ~111 km at any latitude
    kx_de = math.cos(math.radians(float(np.mean(lat_de))))
    kx_ro = math.cos(math.radians(float(np.mean(lat_ro))))
    px = 0.25  # projected-unit padding

    def _rng(lon, lat, kx):
        x = lon * kx
        return (x.min() - px, x.max() + px), (lat.min() - px, lat.max() + px)
    xr_de, yr_de = _rng(lon_de, lat_de, kx_de)
    xr_ro, yr_ro = _rng(lon_ro, lat_ro, kx_ro)
    xs_de, ys_de = xr_de[1] - xr_de[0], yr_de[1] - yr_de[0]
    xs_ro, ys_ro = xr_ro[1] - xr_ro[0], yr_ro[1] - yr_ro[0]

    # size both panels from one inches-per-unit so extents stay true
    figW, L, Rm = 3.5, 0.52, 0.80
    mapW = figW - L - Rm
    scale = mapW / max(xs_de, xs_ro)   # inches per projected unit (both panels)
    W_de, H_de = scale * xs_de, scale * ys_de
    W_ro, H_ro = scale * xs_ro, scale * ys_ro
    top, bot, gap = 0.30, 0.44, 0.36
    figH = top + H_de + gap + H_ro + bot
    fig = plt.figure(figsize=(figW, figH))

    def _ax(w_in, h_in, y0_in):  # left-aligned so the latitude axes line up
        return fig.add_axes([L / figW, y0_in / figH, w_in / figW, h_in / figH])
    ax_de = _ax(W_de, H_de, bot + H_ro + gap)
    ax_ro = _ax(W_ro, H_ro, bot)

    def _draw(ax, lon, lat, st, kx, xr, yr, lon_ticks, title):
        order = np.argsort(st)  # draw main stems over headwaters
        sc = ax.scatter(lon[order] * kx, lat[order], c=st[order],
                        s=0.32 + 0.48 * st[order], cmap="Blues",
                        vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_xticks([t * kx for t in lon_ticks])
        ax.set_xticklabels([str(t) for t in lon_ticks])
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_ylabel("Lat (deg N)", fontsize=7.5)
        ax.tick_params(labelsize=8.4)
        return sc

    _draw(ax_de, lon_de, lat_de, st_de, kx_de, xr_de, yr_de,
          [6, 8, 10, 12, 14], "(a) Germany")
    ax_de.scatter(slon * kx_de, slat, s=11, marker="^", facecolor=RED,
                  edgecolor="white", linewidths=0.25, zorder=5, rasterized=True,
                  label=f"monitored ({len(slon)})")
    ax_de.legend(loc="lower right", fontsize=8.0, markerscale=1.3, framealpha=0.93,
                 handletextpad=0.3, borderpad=0.28)

    sc = _draw(ax_ro, lon_ro, lat_ro, st_ro, kx_ro, xr_ro, yr_ro,
               [20, 22, 24, 26, 28, 30], "(b) Romania")
    ax_ro.set_xlabel("Longitude (deg E)", fontsize=7.5)
    ax_ro.text(0.985, 0.05, "no in-situ labels",
               transform=ax_ro.transAxes, ha="right", va="bottom", fontsize=8.0,
               color="#444444",
               bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", lw=0.5))

    # one colourbar spanning both maps, to the right of the wider panel
    cax = fig.add_axes([(L + max(W_de, W_ro) + 0.10) / figW, bot / figH,
                        0.10 / figW, (H_ro + gap + H_de) / figH])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Strahler stream order (river size)", fontsize=7.5)
    cbar.set_ticks(np.arange(1, int(vmax) + 1))
    cbar.ax.tick_params(labelsize=8.2)
    _save(fig, "fig_studyarea_both.png")


def fig_provenance():
    """Fig. 4: time-aware attribution. (a) Provenance classes of facility-adjacent
    exceedances. (b) How often the flow-defined upstream vs a distance-matched,
    flow-disconnected control set is co-elevated."""
    from matplotlib.lines import Line2D
    ORANGE = "#d95f0e"  # Romania (Germany = BLUE)
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(3.5, 2.72), gridspec_kw={"height_ratios": [1.5, 0.85]})
    fig.subplots_adjust(left=0.30, right=0.965, top=0.965, bottom=0.13, hspace=0.66)

    # (a) provenance classes, horizontal grouped bars
    classes = R.PROVENANCE["classes"]
    y = np.arange(len(classes))[::-1]
    off, h = 0.19, 0.34
    for yi, (label, de, ro) in zip(y, classes):
        axa.barh(yi + off, de, height=h, color=BLUE, zorder=3)
        axa.barh(yi - off, ro, height=h, color=ORANGE, zorder=3)
        axa.text(de + 1.5, yi + off, f"{de}%", va="center", fontsize=7.4, color=BLUE)
        axa.text(ro + 1.5, yi - off, f"{ro}%", va="center", fontsize=7.4, color=ORANGE)
    axa.set_yticks(y)
    axa.set_yticklabels([c[0] for c in classes], fontsize=6.8)
    axa.set_xlim(0, 100)
    axa.set_ylim(-0.62, len(classes) - 0.38)
    axa.set_xlabel("share of classified exceedances (%)", fontsize=7.2)
    axa.tick_params(labelsize=6.6)
    handles = [Line2D([0], [0], color=BLUE, lw=5, label="Germany"),
               Line2D([0], [0], color=ORANGE, lw=5, label="Romania")]
    axa.legend(handles=handles, loc="lower right", fontsize=7.4, frameon=False,
               handlelength=1.1, labelspacing=0.3)
    axa.text(-0.28, 1.02, "(a)", transform=axa.transAxes, fontsize=8.5,
             fontweight="bold", va="bottom")

    # (b) flow-upstream vs distance-matched control, dumbbells per country
    rows = [("Germany", BLUE), ("Romania", ORANGE)]
    for yi, (country, col) in zip([1, 0], rows):
        flow, ctrl, d = R.PROVENANCE["co_elevated"][country]
        axb.plot([min(flow, ctrl), max(flow, ctrl)], [yi] * 2, color=col, lw=1.4,
                 zorder=2)
        axb.plot(flow, yi, "o", color=col, mec="white", mew=0.5, ms=7.5, zorder=3)
        axb.plot(ctrl, yi, "o", mfc="white", mec=col, mew=1.4, ms=7.5, zorder=3)
        axb.text(max(flow, ctrl) + 0.9, yi, f"$\\Delta$ ${d:+.1f}$ pp", va="center",
                 fontsize=6.6, color="#444")
    axb.set_yticks([1, 0])
    axb.set_yticklabels(["Germany", "Romania"], fontsize=7.2)
    axb.set_ylim(-0.55, 3.25)
    axb.set_xlim(75, 91.5)
    axb.set_xticks([75, 80, 85, 90])
    axb.set_xlabel("upstream co-elevated at exceedance (%)", fontsize=7.6)
    axb.tick_params(labelsize=7.4)
    hb = [Line2D([0], [0], marker="o", color="#666", mec="white", mew=0.5, ms=6.5,
                 lw=0, label="flow-defined upstream"),
          Line2D([0], [0], marker="o", mfc="white", mec="#666", mew=1.3, ms=6.5,
                 lw=0, label="distance-matched control")]
    axb.legend(handles=hb, loc="upper left", fontsize=7.4, frameon=False,
               handletextpad=0.3, labelspacing=0.3, borderaxespad=0.1, ncol=1)
    axb.text(-0.28, 1.06, "(b)", transform=axb.transAxes, fontsize=8.5,
             fontweight="bold", va="bottom")
    _save(fig, "fig_provenance.png")


def fig_studyarea_both_bare(print_geometry=False):
    """Fig. 1 inputs: bare study-area rasters plus the geometry for the TikZ wrapper.

    The paper draws Fig. 1's axes, labels and colourbar in TikZ. Only the ~85k reach
    centroids are raster. Each image fills its whole canvas, so image area equals
    data area and TikZ can map pixels to coordinates exactly. Uses the same
    equal-scale projection as fig_studyarea_both(). Returns and prints the geometry.
    """
    import csv as _csv
    import json
    import math
    DATA = os.path.join(os.path.dirname(__file__), "..", "data")

    def _load(country):
        with open(os.path.join(DATA, country, f"segment_graph_{country}.json"),
                  encoding="utf-8") as fh:
            g = json.load(fh)
        lon = np.array([v["lon"] for v in g.values()], float)
        lat = np.array([v["lat"] for v in g.values()], float)
        st = np.array([(v.get("strahler") or 1) for v in g.values()], float)
        return g, lon, lat, st

    gde, lon_de, lat_de, st_de = _load("de")
    gro, lon_ro, lat_ro, st_ro = _load("ro")
    with open(os.path.join(DATA, "de", "strong_labels_national_de.csv"),
              encoding="utf-8") as fh:
        labeled = sorted(set(r["object_id"] for r in _csv.DictReader(fh)))
    slon = np.array([gde[o]["lon"] for o in labeled if o in gde], float)
    slat = np.array([gde[o]["lat"] for o in labeled if o in gde], float)

    vmin, vmax = 0.0, float(max(st_de.max(), st_ro.max()))
    kx_de = math.cos(math.radians(float(np.mean(lat_de))))
    kx_ro = math.cos(math.radians(float(np.mean(lat_ro))))
    px = 0.25

    def _rng(lon, lat, kx):
        x = lon * kx
        return (x.min() - px, x.max() + px), (lat.min() - px, lat.max() + px)
    xr_de, yr_de = _rng(lon_de, lat_de, kx_de)
    xr_ro, yr_ro = _rng(lon_ro, lat_ro, kx_ro)
    xs_de, ys_de = xr_de[1] - xr_de[0], yr_de[1] - yr_de[0]
    xs_ro, ys_ro = xr_ro[1] - xr_ro[0], yr_ro[1] - yr_ro[0]
    mapW = 2.18                       # inches; the TikZ wrapper owns the margins
    scale = mapW / max(xs_de, xs_ro)  # inches per projected unit, shared

    geom = {"vmin": vmin, "vmax": vmax}
    for tag, (lon, lat, st, kx, xr, yr, xs, ys, lon_ticks, extra) in {
        "de": (lon_de, lat_de, st_de, kx_de, xr_de, yr_de, xs_de, ys_de,
               [6, 8, 10, 12, 14], (slon, slat)),
        "ro": (lon_ro, lat_ro, st_ro, kx_ro, xr_ro, yr_ro, xs_ro, ys_ro,
               [20, 22, 24, 26, 28, 30], None),
    }.items():
        w_in, h_in = scale * xs, scale * ys
        fig = plt.figure(figsize=(w_in, h_in))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        order = np.argsort(st)
        ax.scatter(lon[order] * kx, lat[order], c=st[order],
                   s=0.32 + 0.48 * st[order], cmap="Blues",
                   vmin=vmin, vmax=vmax, linewidths=0)
        if extra is not None:
            ax.scatter(extra[0] * kx, extra[1], s=11, marker="^", facecolor=RED,
                       edgecolor="white", linewidths=0.25, zorder=5)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        p = os.path.join(OUT, f"fig_studyarea_{tag}_bare.png")
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(p, dpi=600, pad_inches=0)
        plt.close(fig)
        print("wrote", os.path.relpath(p))
        geom[tag] = {
            "w_cm": round(w_in * 2.54, 4), "h_cm": round(h_in * 2.54, 4),
            "xmin": round(xr[0], 4), "xmax": round(xr[1], 4),
            "ymin": round(yr[0], 4), "ymax": round(yr[1], 4),
            "kx": round(kx, 6),
            "lat_ticks": [int(t) for t in
                          range(int(np.ceil(yr[0])), int(np.floor(yr[1])) + 1)],
            "lon_ticks": [[int(t), round(t * kx, 4)] for t in lon_ticks],
        }
    cmap = plt.get_cmap("Blues")
    geom["cbar_stops"] = [
        [round(v / vmax, 4),
         "".join(f"{int(round(255 * c)):02X}" for c in cmap(v / vmax)[:3])]
        for v in np.linspace(0, vmax, 10)]
    if print_geometry:
        print(json.dumps(geom, indent=1))
    return geom


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", action="store_true",
                    help="also print the Fig. 1 map layout as JSON")
    args = ap.parse_args()
    fig_studyarea_both()
    fig_studyarea_both_bare(print_geometry=args.geometry)
    fig_cofluctuation()
    fig_resolution()
    fig_provenance()
