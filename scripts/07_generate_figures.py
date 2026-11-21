"""Generate Flagship High-Density Multi-Panel Figures (Main Figures 2-7 & Supplementary Figures S1-S5).

Distills the exact visual design grammar of Nature Methods / Nature Biotechnology / Cell (PubCNS):
  - Zero cartoonish icons; crisp #1A1D24 (0.75-0.85 pt) black strokes on all geometric primitives
  - PubCNS 9-swatch scientific color palette (#EB524F/#C92A2A, #4F78C1, #714E9B, #43B04A, #E5A93B, #E3D1A0, #9ECFFF, #F49EB3, #A0A0A0)
  - High-density nested GridSpec hierarchies (7-8 panels per figure, minimal whitespace, clear hero-panel hierarchy)
  - Open-fill boxplots + individual biological replicate scatter + right-angle statistical P-value brackets
  - Dual-sided differential volcano/bubble plots with boxed leader-line callouts (Cell/Cancer Cell style)
  - Dashed ROI niche zoom insets on UMAPs, hierarchical dendrograms + cell-level significance stars (* / **)
  - Smooth genome-browser GRCh38 chromatin profiles + RefSeq exon-intron gene structure models
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, MultipleLocator, NullFormatter
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde, spearmanr
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import FIGURE_EMB_DIR, FIGURE_PRED_DIR
TABLES_DIR = REPO_ROOT / "results" / "tables"
MAIN_FIG_DIR = REPO_ROOT / "figures" / "main"
SUPP_FIG_DIR = REPO_ROOT / "figures" / "supplement"

# High-Contrast Instrument-Grade Scientific Palette (Deep Carmine / Prussian Navy / Royal Violet / Deep Emerald)
INK = "#0F172A"
PUBCNS_DIV = LinearSegmentedColormap.from_list(
    "Deep_Div",
    ["#0B3C5D", "#1D4ED8", "#93C5FD", "#FFFFFF", "#FCA5A5", "#DC2626", "#7F1D1D"],
)
PUBCNS_SEQ = LinearSegmentedColormap.from_list(
    "Deep_Seq",
    ["#F8FAFC", "#FDE68A", "#D97706", "#B91C1C", "#450A0A"],
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.6,
    "axes.titlesize": 9.2,
    "axes.titleweight": "bold",
    "axes.titlepad": 3.6,
    "axes.labelsize": 8.4,
    "axes.labelpad": 2.0,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.85,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.4,
    "ytick.major.size": 3.4,
    "xtick.major.width": 0.80,
    "ytick.major.width": 0.80,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.55,
    "ytick.minor.width": 0.55,
    "xtick.major.pad": 2.4,
    "ytick.major.pad": 2.4,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# High-Contrast Deep Method Identity Palette
PALETTE = {
    "TOLT": "#B91C1C",
    "TOLT_RNA+ATAC": "#B91C1C",
    "TOLT-Base-110": "#B91C1C",
    "TOLT (Base-110)": "#B91C1C",
    "TOLT-Residual": "#B45309",
    "TOLT_RNA": "#1D4ED8",
    "TOLT-Direct": "#1D4ED8",
    "TOLT-F01": "#1E3A8A",
    "F01 Ancestor": "#1E3A8A",
    "totalVI": "#5B21B6",
    "scGPT": "#5B21B6",
    "sciPENN": "#9D174D",
    "scVAEIT_RNA+ATAC": "#6D28D9",
    "scVI-Adapter": "#5B21B6",
    "Seurat-Anchor": "#047857",
    "Geneformer": "#047857",
    "ExtraTrees": "#0F766E",
    "ExtraTrees_RNA+ATAC": "#0F766E",
    "scLinear": "#C2410C",
    "scLinear_RNA": "#C2410C",
    "Cosine-kNN": "#334155",
    "Cosine-kNN_RNA+ATAC": "#334155",
    "SVD-Ridge": "#475569",
    "Classical_RNA+ATAC": "#475569",
    "Classical_RNA": "#64748B",
    "Random-Init": "#94A3B8",
    "TOLT-RandomInit": "#94A3B8",
    "Legacy-Euclid-kNN256": "#94A3B8",
}

LINEAGE_COLORS = [
    "#1D4ED8",  # CD4_Naive_RA (Prussian Blue)
    "#0284C7",  # CD4_Mem_RO (Deep Cerulean)
    "#0F766E",  # CD8_Naive (Deep Teal)
    "#047857",  # CD8_EffMem_KLRG1 (Deep Emerald)
    "#5B21B6",  # B_Naive_IgD (Royal Violet)
    "#9D174D",  # B_Mem_CD27 (Deep Plum)
    "#B91C1C",  # NK_CD56_CD16 (Deep Carmine)
    "#B45309",  # Myeloid_Mono_cDC (Burnt Ochre)
    "#64748B",  # Other (Dark Slate)
]

LINEAGE_SHORT = [
    "CD4 Naive", "CD4 Mem", "CD8 Naive", "CD8 Eff/Mem",
    "B Naive", "B Mem", "NK", "Myeloid", "Other",
]


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    import io
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [
        ("pdf", {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0.02}),
        ("png", {"format": "png", "dpi": 300, "bbox_inches": "tight", "pad_inches": 0.02}),
    ]:
        buf = io.BytesIO()
        fig.savefig(buf, **kw)
        data = buf.getvalue()
        dst = out_dir / f"{name}.{ext}"
        for attempt in range(6):
            try:
                dst.write_bytes(data)
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(0.35)
    plt.close(fig)
    print(f"Saved {out_dir / name}.pdf & .png", flush=True)


def plabel(ax: plt.Axes, letter: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12.8, fontweight="bold", va="bottom", ha="left", color=INK)


def sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str, fontsize: float = 7.4) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.80, color=INK, clip_on=False)
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text(0.5 * (x1 + x2), y + h + 0.007 * yr, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold", color=INK, clip_on=False)


def full_box(ax: plt.Axes, lw: float = 0.85) -> None:
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_edgecolor(INK)


def style_precision_ax(
    ax: plt.Axes,
    *,
    log_x: bool = False,
    cat_x: bool = False,
    cat_y: bool = False,
    grid: str = "both",
) -> None:
    """Apply closed 4-spine frame, inward major/minor precision ticks, and hierarchical scientific gridlines."""
    full_box(ax, 0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", direction="in", top=not cat_x, bottom=True)
    ax.tick_params(axis="y", which="both", direction="in", right=not cat_y, left=True)
    if log_x:
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
    elif not cat_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.xaxis.set_minor_locator( plt.NullLocator() )
    if not cat_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.yaxis.set_minor_locator( plt.NullLocator() )

    if grid in {"both", "y"} and not cat_y:
        ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.45, color="#CBD5E1", alpha=0.85)
        ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.32, color="#E2E8F0", alpha=0.75)
    if grid in {"both", "x"} and not cat_x:
        ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.45, color="#CBD5E1", alpha=0.85)
        if not log_x:
            ax.xaxis.grid(True, which="minor", linestyle=":", linewidth=0.32, color="#E2E8F0", alpha=0.70)


def umap_corner_axes(ax: plt.Axes, xlabel: str = "UMAP1", ylabel: str = "UMAP2") -> None:
    """Draw compact top-journal subtle zero crosshairs and enforce 1:1 isotropic UMAP aspect ratio."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0.0, color="#E2E8F0", ls=":", lw=0.45, zorder=0)
    ax.axvline(0.0, color="#E2E8F0", ls=":", lw=0.45, zorder=0)


# =============================================================================
# FIGURE 2: Mechanistic Dissection of Multi-Omic Pretraining (8 Panels a-h)
# =============================================================================
def make_figure2_mechanisms() -> None:
    df_ab = pd.read_csv(TABLES_DIR / "ablation_backbone_representations.csv")
    df_sw = pd.read_csv(TABLES_DIR / "ablation_simplex_weights.csv")
    df_mc = pd.read_csv(TABLES_DIR / "v41_multiseed_learning_curves.csv")
    df_sub = pd.read_csv(TABLES_DIR / "v41_factorized_subspaces.csv")
    df_cp = pd.read_csv(TABLES_DIR / "gse166489_per_antibody_coupling.csv")
    u_lat = np.load(FIGURE_PRED_DIR / "v41_latent_umap_gse194122.npz", allow_pickle=True)

    fig = plt.figure(figsize=(12.6, 8.14))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[1.0, 0.96, 1.10], hspace=0.32, wspace=0.62)

    # Panel a: 4-Cohort Direct 384-d Probe (Bars + Individual Seed/Replicate Dots + Brackets + Reference Grid)
    ax_a = fig.add_subplot(gs[0, 0:6])
    style_precision_ax(ax_a, cat_x=True, grid="y")
    sub_250 = df_ab[(df_ab["n_ref"] == 250) & (df_ab["readout"].isin(["TOLT-Direct", "SVD-Ridge"]))].copy()
    sub_250 = sub_250[~((sub_250["task"] == "TEAseq_CrossDonor_X066W3_to_X061W1") & (sub_250["modality"] == "RNA"))]
    sub_250["bb_group"] = sub_250["backbone"].replace({
        "rand_s11": "Random-Init", "rand_s29": "Random-Init", "rand_s47": "Random-Init",
        "f01_s29": "F01 Ancestor", "base_110": "TOLT-Base-110", "none": "SVD-Ridge",
    })
    tasks_order = [
        ("BMMC_Stuart_A_to_B", "GSE128639\n(A→B, 25 AB)"),
        ("GSE194122_12710_to_13272", "GSE194122\n(Site2→4, 134 AB)"),
        ("GSE166489_HD12_to_MISC_P1_Expressed", "GSE166489\n(HD→MIS-C)"),
        ("TEAseq_CrossDonor_X066W3_to_X061W1", "TEA-seq\n(X066→X061)"),
    ]
    for ti in range(len(tasks_order)):
        if ti % 2 == 1:
            ax_a.axvspan(ti - 0.48, ti + 0.48, color="#F8FAFC", zorder=0)
        if ti > 0:
            ax_a.axvline(ti - 0.5, color="#94A3B8", ls="--", lw=0.65, zorder=1)
    x = np.arange(len(tasks_order))
    w = 0.195
    cfg_a = [
        ("Random-Init", "#94A3B8", "Random-Init (3 seeds)"),
        ("SVD-Ridge", "#475569", "Local SVD-Ridge"),
        ("F01 Ancestor", "#1D4ED8", "F01-Ancestor (384-d)"),
        ("TOLT-Base-110", "#B91C1C", "TOLT-Base-110 (384-d)"),
    ]
    for bi, (bg, col, lbl) in enumerate(cfg_a):
        means, sds = [], []
        for ti, (tkey, _) in enumerate(tasks_order):
            v = sub_250[(sub_250["task"] == tkey) & (sub_250["bb_group"] == bg)]["protein_pcc_median"].to_numpy(dtype=float)
            means.append(float(v.mean()))
            sds.append(float(v.std()) if len(v) > 1 else 0.0)
            xpos = ti + (bi - 1.5) * w
            if len(v) > 1:
                jit = np.linspace(-0.032, 0.032, len(v))
                ax_a.scatter(xpos + jit, v, s=14, facecolor="white", edgecolor=INK, lw=0.70, zorder=5)
        ax_a.bar(x + (bi - 1.5) * w, means, width=w, yerr=sds if bg == "Random-Init" else None, capsize=2.2,
                 error_kw={"lw": 0.75, "ecolor": INK}, color=col, edgecolor=INK, linewidth=0.80, label=lbl, zorder=3)
    ax_a.axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    for ti, (tkey, _) in enumerate(tasks_order):
        vr = float(sub_250[(sub_250["task"] == tkey) & (sub_250["bb_group"] == "Random-Init")]["protein_pcc_median"].mean())
        vb = float(sub_250[(sub_250["task"] == tkey) & (sub_250["bb_group"] == "TOLT-Base-110")]["protein_pcc_median"].mean())
        sig_bracket(ax_a, ti - 1.5 * w, ti + 1.5 * w, max(vb, vr) + 0.024, 0.020, f"+{vb - vr:.2f}", fontsize=7.8)
    ax_a.set_xticks(x); ax_a.set_xticklabels([t[1] for t in tasks_order], fontweight="bold", fontsize=8.2)
    ax_a.set_ylim(0, 0.96)
    ax_a.set_ylabel("Direct 384-d median PCC ($n_{\\mathrm{ref}}=250$)")
    ax_a.set_title("Frozen 384-d linear probe vs random-init & local SVD")
    ax_a.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, loc="upper right", fontsize=7.5)
    plabel(ax_a, "a", x=-0.10)

    # Panels b & c: Latent UMAPs of Random-Init vs Pretrained TOLT-Base-110 on Unseen Multi-Site BMMC
    ctypes = u_lat["cell_types"].astype(str)
    top_ct = pd.Series(ctypes).value_counts().index[:8].tolist()
    pub_pal = ["#1D4ED8", "#B91C1C", "#047857", "#5B21B6", "#B45309", "#0F766E", "#9D174D", "#0284C7"]
    ct_map = {c: pub_pal[i] for i, c in enumerate(top_ct)}
    pt_cols = [ct_map.get(c, "#CBD5E1") for c in ctypes]

    for ax_u, key_u, ttl_u, ltr, bb_tag in [
        (fig.add_subplot(gs[0, 6:9]), "umap_rand", "Random-Init 384-d UMAP", "b", "Random-Init"),
        (fig.add_subplot(gs[0, 9:12]), "umap_base110", "TOLT-Base-110 384-d UMAP", "c", "TOLT-Base-110"),
    ]:
        xy = u_lat[key_u]
        umap_corner_axes(ax_u)
        ax_u.scatter(xy[:, 0], xy[:, 1], c=pt_cols, s=2.2, alpha=0.90, linewidths=0, rasterized=True, zorder=2)
        ax_u.set_xticks([]); ax_u.set_yticks([])
        full_box(ax_u, 0.85)
        r_sub = df_sub[(df_sub["backbone"] == bb_tag) & (df_sub["subspace"] == "Global_CLS_384d")].iloc[0]
        badge = f"Bio ASW: {r_sub['lineage_bio_asw']:.2f}\nSite Inv.: {r_sub['donor_invariance_score']:.2f}"
        ax_u.text(0.04, 0.95, badge, transform=ax_u.transAxes, va="top", fontsize=7.8, fontweight="bold",
                  bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": INK, "linewidth": 0.70})
        ax_u.set_title(ttl_u, fontsize=9.3)
        ax_u.set_xlabel("UMAP 1 (unseen Site 2 & 4)", fontsize=8.4)
        if ltr == "b":
            ax_u.set_ylabel("UMAP 2", fontsize=8.4)
        plabel(ax_u, ltr, x=-0.11)

    # Panel d: Native 3x128-d Factorized Subspace Specialization Matrix
    ax_d = fig.add_subplot(gs[1, 0:4])
    sub_b = df_sub[df_sub["backbone"] == "TOLT-Base-110"].set_index("subspace")
    sub_r = df_sub[df_sub["backbone"] == "Random-Init"].set_index("subspace")
    row_keys = ["Global_CLS_384d", "Identity_128d", "Modality_Shared_128d", "Modality_Private_128d"]
    row_lbls = ["Global CLS (384d)", "Identity $\\mathbf{z}_{\\mathrm{id}}$ (128d)", "Shared $\\mathbf{z}_{\\mathrm{sh}}$ (128d)", "Private $\\mathbf{z}_{\\mathrm{pr}}$ (128d)"]
    mat_d = np.zeros((5, 3))
    for i, rk in enumerate(row_keys):
        mat_d[i, 0] = float(sub_b.loc[rk, "donor_invariance_score"])
        mat_d[i, 1] = float(sub_b.loc[rk, "lineage_bio_asw"])
        mat_d[i, 2] = float(sub_b.loc[rk, "fewshot_250_pcc_median"])
    mat_d[4, 0] = float(sub_r.loc["Global_CLS_384d", "donor_invariance_score"])
    mat_d[4, 1] = float(sub_r.loc["Global_CLS_384d", "lineage_bio_asw"])
    mat_d[4, 2] = float(sub_r.loc["Global_CLS_384d", "fewshot_250_pcc_median"])
    mat_norm = (mat_d - mat_d.min(0, keepdims=True)) / np.maximum(np.ptp(mat_d, axis=0, keepdims=True), 1e-6)
    sns.heatmap(
        mat_norm, annot=False, cmap=PUBCNS_SEQ, cbar=False,
        xticklabels=["Site\nInvariance", "Lineage\nBio ASW", "Few-Shot\nADT PCC"],
        yticklabels=row_lbls + ["Random-Init (384d)"],
        linewidths=0.80, linecolor=INK, ax=ax_d,
    )
    for ri in range(mat_d.shape[0]):
        for ci in range(mat_d.shape[1]):
            tc = "white" if mat_norm[ri, ci] > 0.62 else INK
            ax_d.text(ci + 0.5, ri + 0.5, f"{mat_d[ri, ci]:.3f}", ha="center", va="center", fontsize=8.4, fontweight="bold", color=tc)
    full_box(ax_d, 0.85)
    ax_d.tick_params(length=0)
    ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0, fontsize=8.0, fontweight="bold")
    ax_d.set_title("3×128-d factorized subspace specialization", fontsize=9.3)
    plabel(ax_d, "d", x=-0.20)

    # Panel e: Pure Inductive 5-Seed Sample-Complexity Ribbon on Held-Out GSE194122 + Crossover Reference Lines
    ax_e = fig.add_subplot(gs[1, 4:8])
    style_precision_ax(ax_e, log_x=True, grid="both")
    mc_90k = df_mc[(df_mc["cohort"] == "GSE194122_BMMC_CrossSite") & (df_mc["method"].isin(["TOLT", "TOLT-F01", "TOLT-Direct", "SVD-Ridge", "TOLT-RandomInit"]))].copy()
    ax_e.axvline(500.0, color="#B91C1C", ls="--", lw=0.85, zorder=2)
    ax_e.text(460.0, 0.34, "$n_{\\mathrm{ref}}^{\\star}\\!=\\!500$", ha="right", va="bottom", fontsize=7.7, fontweight="bold", color="#B91C1C")
    svd_full = float(mc_90k[(mc_90k["method"] == "SVD-Ridge") & (mc_90k["budget"] == "full")]["expressed_pcc_median"].mean())
    ax_e.axhline(svd_full, color="#475569", ls=":", lw=0.80, zorder=2)
    styles_e = [
        ("TOLT", "#B91C1C", "o", "-", "TOLT (Base-110)"),
        ("TOLT-F01", "#1E3A8A", "s", "--", "TOLT (F01-Ancestor)"),
        ("TOLT-Direct", "#1D4ED8", "v", "-", "TOLT-Direct (384-d)"),
        ("SVD-Ridge", "#475569", "D", "-.", "Local SVD-Ridge"),
        ("TOLT-RandomInit", "#94A3B8", "^", ":", "TOLT (Random-Init)"),
    ]
    for mkey, col, mk, ls, lbl in styles_e:
        sub_m = mc_90k[mc_90k["method"] == mkey]
        if not len(sub_m):
            continue
        grp = sub_m.groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), mu=("expressed_pcc_median", "mean"), sd=("expressed_pcc_median", "std")).sort_values("n_ref")
        sd_v = grp["sd"].fillna(0.0).to_numpy()
        ax_e.errorbar(grp["n_ref"], grp["mu"], yerr=sd_v if mkey in {"TOLT", "SVD-Ridge"} else None, capsize=1.8, elinewidth=0.7,
                      marker=mk, linestyle=ls, color=col, lw=2.1 if mkey == "TOLT" else 1.4,
                      markersize=4.8, markeredgecolor=INK, markeredgewidth=0.65, label=lbl, zorder=5 if mkey == "TOLT" else 3)
        if np.any(sd_v > 0):
            ax_e.fill_between(grp["n_ref"], grp["mu"] - sd_v, grp["mu"] + sd_v, color=col, alpha=0.14, zorder=2)
    ax_e.set_xscale("log")
    style_precision_ax(ax_e, log_x=True, grid="both")
    ax_e.set_ylim(0.32, 0.768)
    ax_e.set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$, log scale)")
    ax_e.set_ylabel("Expressed-receptor median PCC")
    ax_e.set_title("5-seed inductive sample complexity", fontsize=9.3)
    ax_e.legend(frameon=True, facecolor="white", edgecolor=INK, loc="lower right", fontsize=7.2)
    plabel(ax_e, "e", x=-0.13)

    # Panel f: OOF Delta^2 Simplex Weight Shift Across Sample Budgets (Pretrained vs Random-Init + w=0.5 Reference Line)
    ax_f_w = fig.add_subplot(gs[1, 8:12])
    style_precision_ax(ax_f_w, cat_x=True, grid="y")
    sw_90k = df_sw[(df_sw["task"] == "GSE194122_12710_to_13272") & (df_sw["backbone"] == "base_110")].sort_values("n_ref")
    sw_rnd = df_sw[(df_sw["task"] == "GSE194122_12710_to_13272") & (df_sw["backbone"].str.startswith("rand_"))].groupby("n_ref", as_index=False)[["mean_w_local_svd", "mean_w_tolt_direct", "mean_w_tolt_residual"]].mean().sort_values("n_ref")
    xw = np.arange(len(sw_90k))
    b_lbls = [str(int(n)) if n < 5000 else "Full" for n in sw_90k["n_ref"]]
    w1 = sw_90k["mean_w_local_svd"].to_numpy()
    w2 = sw_90k["mean_w_tolt_direct"].to_numpy()
    w3 = sw_90k["mean_w_tolt_residual"].to_numpy()
    ax_f_w.axhline(0.50, color=INK, ls="--", lw=0.75, zorder=2)
    ax_f_w.bar(xw - 0.17, w1, width=0.32, color="#64748B", edgecolor=INK, lw=0.70, label="$w_{\\mathrm{loc}}^{\\star}$ (SVD)", zorder=3)
    ax_f_w.bar(xw - 0.17, w2, bottom=w1, width=0.32, color="#1D4ED8", edgecolor=INK, lw=0.70, label="$w_{\\mathrm{dir}}^{\\star}$ (TOLT)", zorder=3)
    ax_f_w.bar(xw - 0.17, w3, bottom=w1 + w2, width=0.32, color="#B91C1C", edgecolor=INK, lw=0.70, label="$w_{\\mathrm{cor}}^{\\star}$ (Resid.)", zorder=3)
    if len(sw_rnd) == len(sw_90k):
        r1 = sw_rnd["mean_w_local_svd"].to_numpy()
        r23 = (sw_rnd["mean_w_tolt_direct"] + sw_rnd["mean_w_tolt_residual"]).to_numpy()
        ax_f_w.bar(xw + 0.17, r1, width=0.32, color="#CBD5E1", edgecolor=INK, lw=0.70, hatch="///", label="Rand. $w_{\\mathrm{loc}}^{\\star}$", zorder=3)
        ax_f_w.bar(xw + 0.17, r23, bottom=r1, width=0.32, color="#FCA5A5", edgecolor=INK, lw=0.70, label="Rand. $w_{\\mathrm{dir+cor}}^{\\star}$", zorder=3)
    ax_f_w.set_xticks(xw); ax_f_w.set_xticklabels(b_lbls, fontweight="bold")
    ax_f_w.set_ylim(0, 1.23)
    ax_f_w.set_xlabel("Reference budget $n_{\\mathrm{ref}}$ (left: TOLT, right: Rand.)")
    ax_f_w.set_ylabel("Mean OOF $\\Delta^2$ simplex weight")
    ax_f_w.set_title("Adaptive OOF prior-to-local weight shift", fontsize=9.3)
    ax_f_w.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=3, fontsize=6.8, loc="upper center", columnspacing=0.7)
    plabel(ax_f_w, "f", x=-0.13)

    # Panel g: Joint Scatter + Black-Bordered Marginal Histograms + Decoupled Rescue Envelope
    gs_g = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2, 0:6], width_ratios=[4.5, 1.0], height_ratios=[1.0, 3.9], hspace=0.05, wspace=0.05)
    ax_g_top = fig.add_subplot(gs_g[0, 0])
    ax_g = fig.add_subplot(gs_g[1, 0])
    ax_g_rt = fig.add_subplot(gs_g[1, 1])
    style_precision_ax(ax_g, grid="both")

    cp_t1 = df_cp[(df_cp["tier"] == "Tier1") & df_cp["cognate_mrna_protein_pcc"].notna()].groupby("protein", as_index=False)[["cognate_mrna_protein_pcc", "tolt_pcc", "svd_pcc", "delta_pcc_tolt_minus_svd", "w_tolt_total"]].mean()
    xg_v = cp_t1["cognate_mrna_protein_pcc"].to_numpy(dtype=float)
    yg_v = cp_t1["delta_pcc_tolt_minus_svd"].to_numpy(dtype=float)
    ax_g.axvline(0.50, color="#1D4ED8", ls="--", lw=0.85, zorder=2)
    ax_g.axhline(0.0, color=INK, linestyle="--", linewidth=0.85, zorder=2)
    ax_g.axhline(0.04, color="#B91C1C", linestyle=":", linewidth=0.80, zorder=2)
    # Upper rescue envelope across bins of cognate mRNA-ADT correlation
    bins_env = np.array([-0.12, 0.22, 0.50, 0.88])
    env_x, env_y = [], []
    for bi_e in range(len(bins_env) - 1):
        msk_e = (xg_v >= bins_env[bi_e]) & (xg_v < bins_env[bi_e + 1])
        if np.any(msk_e):
            env_x.append(0.5 * (bins_env[bi_e] + bins_env[bi_e + 1]))
            env_y.append(float(np.percentile(yg_v[msk_e], 92)))
    ax_g.plot(env_x, env_y, color="#B91C1C", ls="-.", lw=1.20, zorder=2)
    n_top_dec = int(((yg_v >= 0.04) & (xg_v < 0.50)).sum())
    n_top_all = int((yg_v >= 0.04).sum())
    ax_g.text(-0.128, 0.088, f"Decoupled zone ($r_{{\\mathrm{{mRNA}}}}<0.50$)\n{n_top_dec}/{n_top_all} top rescues ($\\Delta$PCC$\\geq 0.04$)", fontsize=7.2, fontweight="bold", color="#7F1D1D",
              bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.65}, zorder=6)
    sc_g = ax_g.scatter(
        xg_v, yg_v, c=cp_t1["tolt_pcc"], cmap=PUBCNS_DIV, vmin=0.35, vmax=0.92,
        s=48, edgecolor=INK, linewidth=0.80, zorder=4,
    )
    offsets_g = {"CD95": (-32, 3), "CD24": (-28, -16), "CD82": (8, 4), "CD45RA": (-46, 2), "CD44": (-34, -2), "CD79b": (8, -11), "CD4": (8, 4)}
    for _, r in cp_t1[cp_t1["protein"].isin(offsets_g)].iterrows():
        pn = str(r["protein"])
        ax_g.annotate(
            pn, (float(r["cognate_mrna_protein_pcc"]), float(r["delta_pcc_tolt_minus_svd"])),
            xytext=offsets_g[pn], textcoords="offset points", fontsize=7.9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": INK, "lw": 0.60},
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.60}, zorder=6,
        )
    ax_g.set_xlim(-0.14, 0.89); ax_g.set_ylim(-0.046, 0.108)
    ax_g.set_xlabel("Cognate mRNA–surface protein correlation ($r_{\\mathrm{mRNA,ADT}}$)")
    ax_g.set_ylabel("$\\Delta$PCC (TOLT $-$ SVD-Ridge)")

    full_box(ax_g_top, 0.80)
    ax_g_top.hist(xg_v, bins=16, color="#1D4ED8", alpha=0.90, edgecolor=INK, linewidth=0.70)
    ax_g_top.axvline(0.50, color="#1D4ED8", ls=":", lw=0.80)
    ax_g_top.set_xlim(ax_g.get_xlim()); ax_g_top.set_xticks([]); ax_g_top.set_yticks([])
    ax_g_top.set_title("Post-transcriptionally decoupled receptors derive maximal gain", fontsize=9.3)
    plabel(ax_g_top, "g", x=-0.10, y=1.08)

    full_box(ax_g_rt, 0.80)
    ax_g_rt.hist(yg_v, bins=13, orientation="horizontal", color="#B91C1C", alpha=0.90, edgecolor=INK, linewidth=0.70)
    ax_g_rt.axhline(0.0, color=INK, linestyle="--", linewidth=0.80)
    ax_g_rt.set_ylim(ax_g.get_ylim()); ax_g_rt.set_xticks([]); ax_g_rt.set_yticks([])
    cb_g = fig.colorbar(sc_g, ax=ax_g_rt, pad=0.08, fraction=0.28)
    cb_g.outline.set_edgecolor(INK); cb_g.outline.set_linewidth(0.75)
    cb_g.set_label("TOLT $r$", fontsize=8.1)

    # Panel h: 3-Stage Dumbbell Escalation (Cognate mRNA r -> SVD-Ridge r -> TOLT r)
    ax_h = fig.add_subplot(gs[2, 7:12])
    style_precision_ax(ax_h, cat_y=True, grid="x")
    showcase = ["CD45RA", "CD95", "CD44", "CD24", "CD82", "CD4", "CD279", "CD38", "KLRG1", "CD14"]
    sub_sh = cp_t1[cp_t1["protein"].isin(showcase)].copy().sort_values("tolt_pcc", ascending=True)
    yh = np.arange(len(sub_sh))
    ax_h.axvline(0.50, color="#64748B", ls=":", lw=0.80, zorder=1)
    for yi, (_, r) in enumerate(sub_sh.iterrows()):
        m_r, s_r, t_r = float(r["cognate_mrna_protein_pcc"]), float(r["svd_pcc"]), float(r["tolt_pcc"])
        if yi % 2 == 0:
            ax_h.axhspan(yi - 0.42, yi + 0.42, color="#F8FAFC", zorder=0)
        ax_h.plot([m_r, t_r], [yi, yi], color=INK, lw=1.25, zorder=2)
        ax_h.scatter([m_r], [yi], color="#B45309", edgecolor=INK, lw=0.80, s=42, marker="D", zorder=3, label="Cognate mRNA $r$" if yi == 0 else "")
        ax_h.scatter([s_r], [yi], color="#1D4ED8", edgecolor=INK, lw=0.80, s=44, marker="s", zorder=4, label="SVD-Ridge $r$" if yi == 0 else "")
        ax_h.scatter([t_r], [yi], color="#B91C1C", edgecolor=INK, lw=0.85, s=56, zorder=5, label="TOLT $r$" if yi == 0 else "")
    ax_h.set_yticks(yh)
    ax_h.set_yticklabels([p.replace("CD279", "PD-1 (CD279)") for p in sub_sh["protein"]], fontweight="bold", fontsize=8.4)
    ax_h.set_xlabel("Cross-donor correlation with surface protein ($r$)")
    ax_h.set_xlim(-0.05, 0.96)
    ax_h.set_title("Escalation beyond cognate mRNA", fontsize=9.3)
    ax_h.legend(frameon=True, facecolor="white", edgecolor=INK, loc="lower right", fontsize=7.5)
    plabel(ax_h, "h", x=-0.16)

    save_fig(fig, MAIN_FIG_DIR, "Figure2")


# =============================================================================
# FIGURE 3: Prior-Likelihood Crossover Law & Global Scorecard (8 Panels a-h)
# =============================================================================
def make_figure3_crossover_scorecard() -> None:
    df_mc = pd.read_csv(TABLES_DIR / "v41_multiseed_learning_curves.csv")
    df_lo = pd.read_csv(TABLES_DIR / "v41_lodo_paired_summary.csv")
    df_sc = pd.read_csv(TABLES_DIR / "v41_global_scorecard.csv")

    fig = plt.figure(figsize=(12.6, 8.41))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[1.0, 0.94, 1.16], hspace=0.33, wspace=1.08)

    focus_m = ["TOLT", "totalVI", "sciPENN", "ExtraTrees", "scLinear", "Seurat-Anchor", "Cosine-kNN", "SVD-Ridge"]

    # Panels a & b: 5-Seed Learning Curves from n_ref=50 to Full with Precision Log Minor Ticks & Crossover Line
    for ax_cur, coh, ycol, ttl, ltr in [
        (fig.add_subplot(gs[0, 0:6]), "GSE194122_BMMC_CrossSite", "expressed_pcc_median", "GSE194122 cross-site BMMC (134 ADTs, expressed $r$)", "a"),
        (fig.add_subplot(gs[0, 6:12]), "GSE128639_BMMC_Stuart", "protein_pcc_median", "GSE128639 cross-donor BMMC (25 ADTs, median $r$)", "b"),
    ]:
        ax_cur.axvline(500.0, color="#B91C1C", ls="--", lw=0.85, zorder=2)
        sub = df_mc[(df_mc["cohort"] == coh) & (df_mc["method"].isin(focus_m))]
        tvi_full = float(sub[(sub["method"] == "totalVI") & (sub["budget"] == "full")][ycol].mean())
        ax_cur.axhline(tvi_full, color="#5B21B6", ls=":", lw=0.80, zorder=2)
        ax_cur.text(52.0, 0.865 if ltr == "b" else 0.772, "Few-shot crossover ($n_{\\mathrm{ref}}\\!=\\!500$)", fontsize=7.3, fontweight="bold", color="#B91C1C", va="top",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.55}, zorder=7)
        for m in focus_m:
            gm = sub[sub["method"] == m].groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), mu=(ycol, "mean"), sd=(ycol, "std")).sort_values("n_ref")
            if not len(gm):
                continue
            lw = 2.3 if m == "TOLT" else 1.4
            mk = "o" if m == "TOLT" else ("s" if m in {"totalVI", "sciPENN"} else "^")
            col = PALETTE[m]
            sd_v = gm["sd"].fillna(0.0).to_numpy()
            ax_cur.errorbar(gm["n_ref"], gm["mu"], yerr=sd_v if m in {"TOLT", "scLinear"} else None, capsize=1.8, elinewidth=0.65,
                            marker=mk, lw=lw, markersize=5.0 if m == "TOLT" else 4.0,
                            color=col, markeredgecolor=INK, markeredgewidth=0.65, label=m, zorder=6 if m == "TOLT" else 3)
            if m in {"TOLT", "scLinear", "SVD-Ridge"} and np.any(sd_v > 0):
                ax_cur.fill_between(gm["n_ref"], gm["mu"] - sd_v, gm["mu"] + sd_v, color=col, alpha=0.14, zorder=2)
        ax_cur.set_xscale("log")
        style_precision_ax(ax_cur, log_x=True, grid="both")
        ax_cur.set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$, log scale)")
        ax_cur.set_ylabel("Cross-donor median PCC")
        ax_cur.set_title(ttl)
        ax_cur.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, loc="lower right", fontsize=7.5)
        plabel(ax_cur, ltr, x=-0.09)

    # Panel c: Panel-Wide MSE Calibration Curves (GSE194122)
    ax_c = fig.add_subplot(gs[1, 0:4])
    ax_c.axvline(500.0, color="#B91C1C", ls="--", lw=0.80, zorder=2)
    for m in ["TOLT", "Seurat-Anchor", "scLinear", "ExtraTrees", "sciPENN", "totalVI"]:
        gm = df_mc[(df_mc["cohort"] == "GSE194122_BMMC_CrossSite") & (df_mc["method"] == m)].groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), mse=("mse", "mean"), sd=("mse", "std")).sort_values("n_ref")
        if len(gm):
            ax_c.plot(gm["n_ref"], gm["mse"], marker="o" if m == "TOLT" else "d", lw=2.2 if m == "TOLT" else 1.35,
                      markersize=4.5, color=PALETTE[m], markeredgecolor=INK, markeredgewidth=0.60, label=m, zorder=5 if m == "TOLT" else 3)
    ax_c.set_xscale("log")
    style_precision_ax(ax_c, log_x=True, grid="both")
    ax_c.set_ylim(0.18, 0.56)
    ax_c.set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$)")
    ax_c.set_ylabel("Panel-wide MSE (lower is better)")
    ax_c.set_title("Global MSE calibration across budgets")
    ax_c.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=7.2, loc="upper right")
    plabel(ax_c, "c", x=-0.14)

    # Panel d: GSE166489 5-Fold LODO Donor-Paired Bars + Error Caps + Replicate Dots
    ax_d = fig.add_subplot(gs[1, 4:8])
    style_precision_ax(ax_d, cat_x=True, grid="y")
    m_lodo = ["TOLT", "scLinear", "ExtraTrees", "Seurat-Anchor", "sciPENN", "totalVI"]
    budgets_d = [("100", "$n_{\\mathrm{ref}}=100$"), ("250", "$n_{\\mathrm{ref}}=250$"), ("6000", "$n_{\\mathrm{ref}}=6{,}000$")]
    xd = np.arange(len(budgets_d))
    wd = 0.132
    for bi in range(len(budgets_d)):
        if bi % 2 == 1:
            ax_d.axvspan(bi - 0.46, bi + 0.46, color="#F8FAFC", zorder=0)
        if bi > 0:
            ax_d.axvline(bi - 0.5, color="#94A3B8", ls="--", lw=0.65)
    for mi, m in enumerate(m_lodo):
        vals, sems = [], []
        for bi, (bcode, _) in enumerate(budgets_d):
            v = df_lo[(df_lo["budget"].astype(str) == bcode) & (df_lo["method"] == m)]["expressed_protein_pcc_median"].to_numpy(dtype=float)
            vals.append(float(v.mean()))
            sems.append(float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
            xpos = bi + (mi - 2.5) * wd
            if m in {"TOLT", "totalVI"}:
                jit = np.linspace(-0.022, 0.022, len(v))
                ax_d.scatter(xpos + jit, v, s=11, facecolor="white", edgecolor=INK, lw=0.60, zorder=5)
        ax_d.bar(xd + (mi - 2.5) * wd, vals, width=wd, yerr=sems, capsize=1.6, error_kw={"lw": 0.65, "ecolor": INK},
                 color=PALETTE[m], edgecolor=INK, linewidth=0.72, label=m, zorder=3)
    ax_d.axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    sig_bracket(ax_d, 0 - 2.5 * wd, 0 + 2.5 * wd, 0.615, 0.022, "+0.12 vs totalVI", fontsize=7.4)
    ax_d.set_xticks(xd); ax_d.set_xticklabels([b[1] for b in budgets_d], fontweight="bold")
    ax_d.set_ylim(0.28, 0.72)
    ax_d.set_ylabel("5-donor LODO expressed PCC")
    ax_d.set_title("GSE166489 188-AB 5-fold LODO")
    ax_d.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=7.0, loc="lower right")
    plabel(ax_d, "d", x=-0.14)

    # Panel e: Laptop Compute-Accuracy Pareto Frontier with Stepped Envelope & Latency Reference Line
    ax_e = fig.add_subplot(gs[1, 8:12])
    par_m = ["TOLT", "TOLT-Direct", "totalVI", "sciPENN", "ExtraTrees", "scLinear", "Cosine-kNN", "Seurat-Anchor", "SVD-Ridge"]
    sub_p = df_mc[(df_mc["cohort"].isin(["GSE194122_BMMC_CrossSite", "GSE128639_BMMC_Stuart"])) & (df_mc["budget"] == "250") & (df_mc["method"].isin(par_m))]
    par_g = sub_p.groupby(["cohort", "method"], as_index=False).agg(sec=("fit_predict_sec", "mean"), pcc=("protein_pcc_median", "mean"), vram=("peak_vram_mb", "max")).groupby("method", as_index=False).agg(sec=("sec", "mean"), pcc=("pcc", "mean"), vram=("vram", "max"))

    pt_dir = par_g[par_g["method"] == "TOLT-Direct"].iloc[0]
    pt_tolt = par_g[par_g["method"] == "TOLT"].iloc[0]
    ax_e.axvline(10.0, color="#64748B", ls=":", lw=0.75, zorder=1)
    ax_e.fill_between([max(float(pt_dir["sec"]), 0.04), float(pt_tolt["sec"]), 320.0],
                      [float(pt_dir["pcc"]), float(pt_tolt["pcc"]), float(pt_tolt["pcc"])],
                      0.52, color="#FEE2E2", alpha=0.45, zorder=1)
    ax_e.plot([max(float(pt_dir["sec"]), 0.04), max(float(pt_dir["sec"]), 0.04), float(pt_tolt["sec"]), 320.0],
              [0.52, float(pt_dir["pcc"]), float(pt_tolt["pcc"]), float(pt_tolt["pcc"])],
              color="#B91C1C", linestyle="--", lw=1.35, zorder=2)

    off_e = {"TOLT": (6, 3), "TOLT-Direct": (6, 4), "ExtraTrees": (6, -8), "scLinear": (-44, -2), "Cosine-kNN": (-52, -9), "totalVI": (-40, 7), "SVD-Ridge": (-48, 1), "Seurat-Anchor": (6, -8), "sciPENN": (6, 2)}
    for _, r in par_g.iterrows():
        m = str(r["method"])
        sx = max(float(r["sec"]), 0.04)
        sy = float(r["pcc"])
        sz = 82 + 0.14 * float(r["vram"])
        ax_e.scatter(sx, sy, s=sz, color=PALETTE[m], edgecolor=INK, linewidth=0.90, zorder=4)
        dx, dy = off_e.get(m, (5, 3))
        ax_e.annotate(m, (sx, sy), xytext=(dx, dy), textcoords="offset points", fontsize=8.0, fontweight="bold" if m.startswith("TOLT") else "normal")
    if "totalVI" in set(par_g["method"]):
        pt_tvi = par_g[par_g["method"] == "totalVI"].iloc[0]
        spd = float(pt_tvi["sec"]) / max(float(pt_tolt["sec"]), 0.1)
        ax_e.annotate("", xy=(float(pt_tolt["sec"]), 0.608), xytext=(float(pt_tvi["sec"]), 0.608),
                      arrowprops={"arrowstyle": "<->", "color": "#B91C1C", "lw": 1.20})
        ax_e.text(np.sqrt(float(pt_tolt["sec"]) * float(pt_tvi["sec"])), 0.614, f"{spd:.0f}× faster", ha="center", fontsize=8.2, fontweight="bold", color="#B91C1C")
    ax_e.set_xscale("log")
    style_precision_ax(ax_e, log_x=True, grid="both")
    ax_e.set_xlim(0.03, 360); ax_e.set_ylim(0.525, 0.692)
    ax_e.set_xlabel("Adaptation + inference wall-clock time (s)")
    ax_e.set_ylabel("Mean PCC ($n_{\\mathrm{ref}}=250$)")
    ax_e.set_title("Compute–accuracy Pareto (RTX 5070 8 GB)")
    plabel(ax_e, "e", x=-0.14)

    # Panels f, g, h: scIB-Style Global Multi-Cohort Scorecard + Composite Rank + Error Bars
    gs_fgh = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, :], width_ratios=[4.35, 1.02, 0.95], wspace=0.05)
    ax_f = fig.add_subplot(gs_fgh[0, 0])
    ax_g = fig.add_subplot(gs_fgh[0, 1])
    ax_h = fig.add_subplot(gs_fgh[0, 2])

    df_sc = df_sc.sort_values("Overall_Composite_Score", ascending=True).reset_index(drop=True)
    m_names = df_sc["method"].tolist()
    metric_cols = [
        ("FewShot_CrossSite_PCC", "Few-Shot\nCross-Site r"),
        ("FewShot_CrossDonor_PCC", "Few-Shot\nDonor r"),
        ("FewShot_188AB_LODO_PCC", "Few-Shot\n188-AB r"),
        ("Atlas_Expressed_PCC", "Full-Atlas\nExpressed r"),
        ("Calibration_NegMSE", "Calibration\n(1 − MSE)"),
        ("Bimodal_GMM_AUROC", "2-GMM Gate\nAUROC"),
        ("Manifold_Leiden_NMI", "Surface State\nLeiden NMI"),
        ("Speed_LogInvSec", "Laptop Speed\nScore"),
    ]
    ax_f.set_facecolor("#F8FAFC")
    for ci, (cname, _) in enumerate(metric_cols):
        vals = df_sc[cname].to_numpy(dtype=float)
        med_v = float(np.nanmedian(vals))
        vals_f = np.where(np.isnan(vals), med_v, vals)
        n_v = (vals_f - vals_f.min()) / max(float(np.ptp(vals_f)), 1e-8)
        for ri in range(len(m_names)):
            sz = 155 + 155 * n_v[ri]
            ax_f.scatter(ci, ri, s=sz, c=[n_v[ri]], cmap=PUBCNS_DIV, vmin=0.0, vmax=1.0, edgecolor=INK, linewidth=0.80, zorder=3)
            disp_val = vals_f[ri]
            txt_c = "white" if (n_v[ri] > 0.72 or n_v[ri] < 0.24) else INK
            ax_f.text(ci, ri, f"{disp_val:.2f}", ha="center", va="center", fontsize=7.1, fontweight="bold", color=txt_c, zorder=4)
    ax_f.set_xticks(np.arange(len(metric_cols)))
    ax_f.set_xticklabels([m[1] for m in metric_cols], fontsize=8.4, fontweight="bold")
    ax_f.set_yticks(np.arange(len(m_names)))
    ax_f.set_yticklabels(m_names, fontsize=8.9, fontweight="bold")
    ax_f.set_xlim(-0.55, len(metric_cols) - 0.45); ax_f.set_ylim(-0.6, len(m_names) - 0.4)
    ax_f.grid(True, linestyle="--", linewidth=0.45, color="#CBD5E1", zorder=1)
    full_box(ax_f, 0.85)
    ax_f.set_title("Multi-cohort benchmark scorecard across 8 evaluation dimensions")
    plabel(ax_f, "f", x=-0.08)

    y_pos = np.arange(len(m_names))
    bar_cols = [PALETTE.get(m, "#475569") for m in m_names]
    style_precision_ax(ax_g, cat_y=True, grid="x")
    tolt_comp = float(df_sc.loc[df_sc["method"] == "TOLT", "Overall_Composite_Score"].iloc[0])
    ax_g.axvline(tolt_comp, color="#B91C1C", ls="--", lw=0.80, zorder=2)
    ax_g.barh(y_pos, df_sc["Overall_Composite_Score"], height=0.66, color=bar_cols, edgecolor=INK, linewidth=0.80, zorder=3)
    for ri, sc_v in enumerate(df_sc["Overall_Composite_Score"]):
        ax_g.text(float(sc_v) + 0.015, ri, f"{sc_v:.3f}", va="center", fontsize=7.8, fontweight="bold",
                  bbox={"boxstyle": "square,pad=0.04", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}, zorder=5)
    ax_g.set_yticks(y_pos); ax_g.set_yticklabels([])
    ax_g.set_ylim(-0.6, len(m_names) - 0.4); ax_g.set_xlim(0, 1.22)
    ax_g.set_xlabel("Composite score", fontweight="bold")
    ax_g.set_title("Overall rank", fontsize=9.3)
    plabel(ax_g, "g", x=-0.08)

    # Panel h: Few-shot Calibration Error (MSE at n_ref=250) per Method
    style_precision_ax(ax_h, cat_y=True, grid="x")
    mse_map = df_mc[(df_mc["cohort"] == "GSE128639_BMMC_Stuart") & (df_mc["budget"] == "250")].groupby("method")["mse"].mean().to_dict()
    mse_vals = [float(mse_map.get(m, 0.25)) for m in m_names]
    tolt_mse = mse_vals[m_names.index("TOLT")] if "TOLT" in m_names else min(mse_vals)
    ax_h.axvline(tolt_mse, color="#B91C1C", ls="--", lw=0.80, zorder=2)
    ax_h.barh(y_pos, mse_vals, height=0.66, color="#64748B", edgecolor=INK, linewidth=0.75, zorder=3)
    for ri, mv in enumerate(mse_vals):
        if m_names[ri] == "TOLT":
            ax_h.patches[ri].set_facecolor("#B91C1C")
        ax_h.text(mv + 0.008, ri, f"{mv:.3f}", va="center", fontsize=7.5, fontweight="bold",
                  bbox={"boxstyle": "square,pad=0.04", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}, zorder=5)
    ax_h.set_yticks(y_pos); ax_h.set_yticklabels([])
    ax_h.set_ylim(-0.6, len(m_names) - 0.4); ax_h.set_xlim(0, 0.52)
    ax_h.set_xlabel("Few-shot MSE ↓", fontweight="bold")
    ax_h.set_title("Error ($n=250$)", fontsize=9.3)
    plabel(ax_h, "h", x=-0.08)

    save_fig(fig, MAIN_FIG_DIR, "Figure3")


# =============================================================================
# FIGURE 4: Compositional Hygiene, 1D 2-GMM Curves & Manifold Recovery (a-h)
# =============================================================================
def make_figure4_hygiene_gmm_manifold() -> None:
    df_st = pd.read_csv(TABLES_DIR / "gse166489_antibody_stratification.csv")
    df_cp = pd.read_csv(TABLES_DIR / "gse166489_per_antibody_coupling.csv")
    df_gm = pd.read_csv(TABLES_DIR / "gse166489_bimodal_gmm_gating.csv")
    df_mn = pd.read_csv(TABLES_DIR / "v41_tier1_manifold_recovery.csv")
    gmm_npz = np.load(FIGURE_PRED_DIR / "v41_gmm_densities_and_roc.npz", allow_pickle=True)
    man_u = np.load(FIGURE_PRED_DIR / "v41_manifold_umaps_P1.1.npz", allow_pickle=True)
    bd_hd3 = np.load((FIGURE_EMB_DIR / "gse166489_A.HD3_bundle.npz"), allow_pickle=True)

    fig = plt.figure(figsize=(12.51, 8.14))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[0.96, 1.0, 1.02], hspace=0.33, wspace=0.82)

    # Panel a: 188-Antibody Reference SD vs 99th-Percentile Phase Diagram
    ax_a = fig.add_subplot(gs[0, 0:4])
    style_precision_ax(ax_a, grid="both")
    y_hd3 = np.log1p(bd_hd3["adt_counts"].astype(np.float32) / np.exp(np.log1p(bd_hd3["adt_counts"].astype(np.float32)).mean(1, keepdims=True)))
    pnames = [str(x) for x in bd_hd3["adt_names"]]
    sd_p = y_hd3.std(0)
    q99_p = np.percentile(y_hd3, 99, axis=0)
    t1 = (sd_p >= 0.35) | (q99_p >= 1.50)
    t2 = (~t1) & ((sd_p >= 0.22) | (q99_p >= 1.00))
    t3 = (~t1) & (~t2)
    for mask_t, col_t, lbl_t in [
        (t1, "#B91C1C", "Tier 1 Expressed (LODO n=95)"),
        (t2, "#B45309", "Tier 2 Rare/Induc. (LODO n=57)"),
        (t3, "#475569", "Tier 3 Background (LODO n=36)"),
    ]:
        ax_a.scatter(sd_p[mask_t], q99_p[mask_t], s=28, color=col_t, edgecolor=INK, linewidth=0.65, alpha=0.92, label=lbl_t, zorder=3)
    ax_a.axvline(0.35, color="#B91C1C", ls="--", lw=0.85, zorder=2); ax_a.axhline(1.50, color="#B91C1C", ls="--", lw=0.85, zorder=2)
    ax_a.axvline(0.22, color="#B45309", ls=":", lw=0.75, zorder=2); ax_a.axhline(1.00, color="#B45309", ls=":", lw=0.75, zorder=2)
    off_4a = {"CD3": (-26, -13), "CD4": (-12, 8), "CD19": (-28, 7), "CD14": (14, -12), "HLADR": (-38, 6), "CD326": (16, -8)}
    for mk_lbl, xy_off in off_4a.items():
        if mk_lbl in pnames:
            ji = pnames.index(mk_lbl)
            disp_m = "HLA-DR" if mk_lbl == "HLADR" else ("EpCAM" if mk_lbl == "CD326" else mk_lbl)
            ax_a.annotate(
                disp_m, (float(sd_p[ji]), float(q99_p[ji])), xytext=xy_off, textcoords="offset points",
                fontsize=7.4, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": INK, "lw": 0.55},
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.55}, zorder=6,
            )
    ax_a.set_xlabel("Reference CLR standard deviation ($\\sigma_{\\mathrm{ref}}$)")
    ax_a.set_ylabel("Reference 99th-percentile CLR")
    ax_a.set_title("188-antibody abundance phase space")
    ax_a.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.3, loc="upper left")
    plabel(ax_a, "a", x=-0.14)

    # Panel b: Open-Fill Boxplot + Individual Antibody Scatter Across Tiers 1-3 + Zero Reference Line
    ax_b = fig.add_subplot(gs[0, 4:8])
    style_precision_ax(ax_b, cat_x=True, grid="y")
    cp_ab = df_cp.groupby(["protein", "tier"], as_index=False)["tolt_pcc"].mean()
    tier_ord = ["Tier1", "Tier2", "Tier3"]
    tier_cols = {"Tier1": "#B91C1C", "Tier2": "#B45309", "Tier3": "#475569"}
    ax_b.axhline(0.0, color=INK, ls="--", lw=0.75, zorder=2)
    ax_b.axhline(0.50, color="#1D4ED8", ls=":", lw=0.75, zorder=2)
    for ti, tk in enumerate(tier_ord):
        if ti % 2 == 1:
            ax_b.axvspan(ti - 0.45, ti + 0.45, color="#F8FAFC", zorder=0)
        vals_t = cp_ab[cp_ab["tier"] == tk]["tolt_pcc"].to_numpy(dtype=float)
        bp = ax_b.boxplot([vals_t], positions=[ti], widths=0.46, patch_artist=True, showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor("white"); patch.set_edgecolor(INK); patch.set_linewidth(0.85)
        for med in bp["medians"]:
            med.set_color(tier_cols[tk]); med.set_linewidth(1.7)
        rng = np.random.default_rng(100 + ti)
        jit = rng.uniform(-0.15, 0.15, size=len(vals_t))
        ax_b.scatter(np.full_like(vals_t, ti) + jit, vals_t, s=18, color=tier_cols[tk], edgecolor=INK, lw=0.50, alpha=0.88, zorder=4)
    sig_bracket(ax_b, 0, 2, 0.98, 0.04, "$P < 10^{-16}$ (MWU)", fontsize=7.8)
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["Tier 1 Expressed\n(n=95, med 0.65)", "Tier 2 Rare/Induc.\n(n=57, med 0.18)", "Tier 3 Epithelial\n(n=36, med 0.07)"], fontweight="bold", fontsize=7.9)
    ax_b.set_ylim(-0.12, 1.12)
    ax_b.set_ylabel("5-donor LODO Pearson $r$")
    ax_b.set_title("Per-receptor fidelity across objective tiers")
    plabel(ax_b, "b", x=-0.14)

    # Panel c: 2x2 Sub-Grid of Physical 1D Bimodal Gaussian Mixture Gates vs Broken q75 Cutoff
    gs_c = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 8:12], hspace=0.30, wspace=0.20)
    for idx_m, mk in enumerate(["CD19", "IgD", "CD14", "CD3"]):
        ax_m = fig.add_subplot(gs_c[idx_m // 2, idx_m % 2])
        style_precision_ax(ax_m, cat_y=True, grid="x")
        te_v = gmm_npz[f"{mk}_te_vals"]
        xs = gmm_npz[f"{mk}_xs"]
        p_neg = gmm_npz[f"{mk}_pdf_neg"]
        p_pos = gmm_npz[f"{mk}_pdf_pos"]
        thr_gmm, thr_q75 = float(gmm_npz[f"{mk}_thr"][0]), float(gmm_npz[f"{mk}_thr"][1])
        ax_m.hist(te_v[te_v > 0.02], bins=28, density=True, color="#CBD5E1", edgecolor=INK, linewidth=0.45, alpha=0.90, zorder=2)
        ax_m.plot(xs, p_neg, color="#1D4ED8", lw=1.55, label="$\\mathcal{N}_-$" if idx_m == 0 else "", zorder=3)
        ax_m.plot(xs, p_pos, color="#B91C1C", lw=1.65, label="$\\mathcal{N}_+$" if idx_m == 0 else "", zorder=3)
        ax_m.axvline(thr_q75, color="#334155", ls=":", lw=1.40, label="q75" if idx_m == 0 else "", zorder=4)
        ax_m.axvline(thr_gmm, color="#B91C1C", ls="-", lw=1.55, label="$\\tau_p^{\\star}$" if idx_m == 0 else "", zorder=4)
        pos_pct = 100.0 * float((te_v >= thr_gmm).mean())
        ax_m.set_title(f"{mk} ({pos_pct:.1f}% pos.)", fontsize=9.2)
        ax_m.set_yticks([])
        if idx_m >= 2:
            ax_m.set_xlabel("Measured CLR", fontsize=8.4)
        if idx_m == 0:
            ax_m.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.5, loc="upper right")
            plabel(ax_m, "c", x=-0.16, y=1.10)

    # Panel d: Dumbbell Rescue (Legacy q75 vs 2-GMM AUROC + Reference Threshold)
    ax_d = fig.add_subplot(gs[1, 0:4])
    style_precision_ax(ax_d, cat_y=True, grid="x")
    gm_t = df_gm[df_gm["method"] == "TOLT"].groupby("marker", as_index=False)[["legacy_q75_auroc", "gmm_auroc"]].mean()
    mk_ord = [m for m in ["CD19", "IgD", "CD14", "CD56", "CD279", "CD3", "HLADR", "CD8a", "CD4"] if m in set(gm_t["marker"])]
    gm_t = gm_t.set_index("marker").loc[mk_ord].reset_index()
    yd = np.arange(len(mk_ord))
    ax_d.axvline(0.90, color="#1D4ED8", ls="--", lw=0.80, zorder=1)
    for yi, (_, r) in enumerate(gm_t.iterrows()):
        q_v, g_v = float(r["legacy_q75_auroc"]), float(r["gmm_auroc"])
        if yi % 2 == 0:
            ax_d.axhspan(yi - 0.42, yi + 0.42, color="#F8FAFC", zorder=0)
        ax_d.plot([q_v, g_v], [yi, yi], color=INK, lw=1.25, zorder=2)
        ax_d.scatter([q_v], [yi], color="#64748B", edgecolor=INK, lw=0.80, s=44, marker="s", zorder=3, label="Fixed q75" if yi == 0 else "")
        ax_d.scatter([g_v], [yi], color="#B91C1C", edgecolor=INK, lw=0.85, s=54, zorder=4, label="2-GMM $\\tau_p^{\\star}$" if yi == 0 else "")
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([m.replace("HLADR", "HLA-DR").replace("CD279", "PD-1") for m in mk_ord], fontweight="bold")
    ax_d.set_xlim(0.62, 1.02)
    ax_d.set_xlabel("5-donor LODO gating AUROC")
    ax_d.set_title("Bayesian 2-GMM rescues rare & pan-T gates")
    ax_d.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.4, loc="lower right")
    plabel(ax_d, "d", x=-0.14)

    # Panel e: Precision-Recall Curves on Rare B-Cell Gate (CD19) with Precision Grid & Reference Line
    ax_e = fig.add_subplot(gs[1, 4:8])
    style_precision_ax(ax_e, grid="both")
    ax_e.axhline(0.80, color="#64748B", ls=":", lw=0.75, zorder=2)
    for m in ["TOLT", "scLinear", "Seurat-Anchor", "totalVI", "sciPENN"]:
        k_pr = f"pr_CD19_{m}"
        if k_pr in gmm_npz.files:
            rc, pr = gmm_npz[k_pr][0], gmm_npz[k_pr][1]
            ap_v = float(df_gm[(df_gm["marker"] == "CD19") & (df_gm["method"] == m)]["gmm_auprc"].mean())
            ax_e.plot(rc, pr, lw=2.3 if m == "TOLT" else 1.35, color=PALETTE[m], label=f"{m} ({ap_v:.3f})", zorder=5 if m == "TOLT" else 3)
            if m == "TOLT":
                ax_e.fill_between(rc, 0.45, pr, color="#B91C1C", alpha=0.12, zorder=1)
    ax_e.set_xlim(0, 1.02); ax_e.set_ylim(0.45, 1.02)
    ax_e.set_xlabel("Recall (rare $\\mathrm{CD19^+}$ B cells, ~9.5%)")
    ax_e.set_ylabel("Precision")
    ax_e.set_title("Cross-donor precision–recall (CD19)")
    ax_e.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.3, loc="lower left")
    plabel(ax_e, "e", x=-0.14)

    # Panel f: High-Contrast Cross-Method 2-GMM AUROC Heatmap
    ax_f = fig.add_subplot(gs[2, 0:4])
    piv_f = df_gm[df_gm["method"].isin(["TOLT", "scLinear", "Seurat-Anchor", "totalVI", "sciPENN"])].groupby(["marker", "method"], as_index=False)["gmm_auroc"].mean().pivot(index="marker", columns="method", values="gmm_auroc")
    cols_f = ["TOLT", "scLinear", "Seurat-Anchor", "totalVI", "sciPENN"]
    rows_f = [m for m in ["CD19", "CD3", "CD4", "CD8a", "HLADR", "IgD", "CD56", "CD279", "CD14", "CD38"] if m in piv_f.index]
    disp_r = [m.replace("HLADR", "HLA-DR").replace("CD279", "PD-1") for m in rows_f]
    mat_f = piv_f.loc[rows_f, cols_f].to_numpy(dtype=float)
    sns.heatmap(piv_f.loc[rows_f, cols_f], annot=False, cmap=PUBCNS_DIV, vmin=0.81, vmax=0.992,
                yticklabels=disp_r, linewidths=0.75, linecolor=INK, cbar_kws={"label": "2-GMM AUROC", "pad": 0.02, "fraction": 0.038},
                ax=ax_f)
    for ri in range(mat_f.shape[0]):
        for ci in range(mat_f.shape[1]):
            nv = (mat_f[ri, ci] - 0.81) / (0.992 - 0.81)
            tc = "white" if (nv > 0.72 or nv < 0.22) else INK
            ax_f.text(ci + 0.5, ri + 0.5, f"{mat_f[ri, ci]:.3f}", ha="center", va="center", fontsize=7.4, fontweight="bold", color=tc)
    full_box(ax_f, 0.85)
    ax_f.tick_params(length=0)
    ax_f.set_xticklabels(cols_f, rotation=24, ha="right", fontsize=7.7, fontweight="bold")
    ax_f.set_yticklabels(disp_r, rotation=0, fontweight="bold", fontsize=8.0)
    ax_f.set_xlabel(""); ax_f.set_ylabel("")
    ax_f.set_title("5-donor LODO 2-GMM AUROC matrix")
    plabel(ax_f, "f", x=-0.15)

    # Panels g & h: Tier-1 Signal-Space Surface Manifold UMAP & Leiden ARI/NMI Recovery
    gs_gh = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 4:8], wspace=0.07)
    ct_str = man_u["true_leiden"].astype(str)
    uniq_c = sorted(np.unique(ct_str), key=lambda z: int(z) if z.isdigit() else 99)
    pal_c = ["#1D4ED8", "#B91C1C", "#047857", "#5B21B6", "#B45309", "#0284C7", "#9D174D", "#0F766E", "#C2410C", "#334155", "#6D28D9", "#1E3A8A"]
    c_lookup = {k: pal_c[i % len(pal_c)] for i, k in enumerate(uniq_c)}
    c_pts = [c_lookup[k] for k in ct_str]
    for sub_i, (ukey, uttl) in enumerate([("umap_measured", "Measured ADT"), ("umap_TOLT", "TOLT Few-Shot")]):
        ax_ug = fig.add_subplot(gs_gh[0, sub_i])
        umap_corner_axes(ax_ug)
        xy = man_u[ukey]
        ax_ug.scatter(xy[:, 0], xy[:, 1], c=c_pts, s=1.4, alpha=0.90, linewidths=0, rasterized=True, zorder=2)
        ax_ug.set_xticks([]); ax_ug.set_yticks([])
        full_box(ax_ug, 0.80)
        ax_ug.set_title(uttl, fontsize=9.3)
        ax_ug.set_xlabel("Unseen P1.1", fontsize=7.6, fontweight="bold")
        if sub_i == 0:
            plabel(ax_ug, "g", x=-0.10)

    ax_h = fig.add_subplot(gs[2, 8:12])
    style_precision_ax(ax_h, cat_x=True, grid="y")
    mn_fs = df_mn[(df_mn["cohort"] == "GSE166489_Tier1_SurfaceManifold") & (df_mn["budget"] == "ref6000")].groupby("method", as_index=False)[["leiden_ari", "leiden_nmi"]].agg(["mean", "std"])
    mn_fs.columns = ["method", "ari_mu", "ari_sd", "nmi_mu", "nmi_sd"] if "method" in mn_fs.columns else mn_fs.columns
    mn_raw = df_mn[(df_mn["cohort"] == "GSE166489_Tier1_SurfaceManifold") & (df_mn["budget"] == "ref6000")]
    m_ord_h = [m for m in ["TOLT", "scLinear", "TOLT-Residual", "totalVI", "sciPENN", "Seurat-Anchor", "ExtraTrees"] if m in set(mn_raw["method"])]
    ari_m = [float(mn_raw[mn_raw["method"] == m]["leiden_ari"].mean()) for m in m_ord_h]
    ari_s = [float(mn_raw[mn_raw["method"] == m]["leiden_ari"].std(ddof=1) / np.sqrt(5)) for m in m_ord_h]
    nmi_m = [float(mn_raw[mn_raw["method"] == m]["leiden_nmi"].mean()) for m in m_ord_h]
    nmi_s = [float(mn_raw[mn_raw["method"] == m]["leiden_nmi"].std(ddof=1) / np.sqrt(5)) for m in m_ord_h]
    xh = np.arange(len(m_ord_h))
    ax_h.axhline(ari_m[0], color="#1D4ED8", ls="--", lw=0.80, zorder=2)
    ax_h.bar(xh - 0.18, ari_m, width=0.35, yerr=ari_s, capsize=1.8, error_kw={"lw": 0.65, "ecolor": INK}, color="#1D4ED8", edgecolor=INK, lw=0.80, label="Leiden ARI (#1 TOLT)", zorder=3)
    ax_h.bar(xh + 0.18, nmi_m, width=0.35, yerr=nmi_s, capsize=1.8, error_kw={"lw": 0.65, "ecolor": INK}, color="#B91C1C", edgecolor=INK, lw=0.80, label="Leiden NMI", zorder=3)
    ax_h.set_xticks(xh); ax_h.set_xticklabels(m_ord_h, rotation=24, ha="right", fontsize=8.0, fontweight="bold")
    ax_h.set_ylim(0.38, 0.82)
    ax_h.set_ylabel("Tier-1 manifold agreement")
    ax_h.set_title("5-donor LODO surface manifold recovery")
    ax_h.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=7.5, loc="upper right")
    plabel(ax_h, "h", x=-0.14)

    save_fig(fig, MAIN_FIG_DIR, "Figure4")


# =============================================================================
# FIGURE 5: 18-Donor (162,777-Cell) MIS-C Atlas, Volcano Callouts & Clustered Heatmap (a-g)
# =============================================================================
def make_figure5_misc_18donor_atlas() -> None:
    atlas = np.load(FIGURE_PRED_DIR / "v41_misc_18donor_atlas.npz", allow_pickle=True)
    df_path = pd.read_csv(TABLES_DIR / "gse166489_18donor_misc_pathology.csv")
    df_diff = pd.read_csv(TABLES_DIR / "v41_misc_differential_receptors.csv")
    df_hm = pd.read_csv(TABLES_DIR / "v41_misc_18x40_heatmap.csv")
    u_p1 = np.load(FIGURE_PRED_DIR / "umap_coords_P1.1.npz", allow_pickle=True)

    fig = plt.figure(figsize=(12.69, 10.18))
    gs = gridspec.GridSpec(4, 12, figure=fig, height_ratios=[1.00, 0.70, 1.04, 1.40], hspace=0.33, wspace=0.78)

    # Panel a: 18-Donor (162,777-cell) Integrated PBMC UMAP + Dashed ROI Zoom Inset on Myeloid Island
    ax_a = fig.add_subplot(gs[0, 0:5])
    umap_corner_axes(ax_a)
    u_xy = atlas["umap_xy"]
    lin = atlas["lineages"]
    cond_arr = atlas["conditions"]
    pt_c = [LINEAGE_COLORS[int(i)] for i in lin]
    ax_a.scatter(u_xy[:, 0], u_xy[:, 1], c=pt_c, s=1.1, alpha=0.88, linewidths=0, rasterized=True, zorder=2)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    full_box(ax_a, 0.85)

    my_mask = lin == 7
    if my_mask.sum() > 50:
        mx0, my0 = float(np.median(u_xy[my_mask, 0])), float(np.median(u_xy[my_mask, 1]))
        rx, ry = 2.4, 2.4
        ax_ins = ax_a.inset_axes([0.64, 0.51, 0.33, 0.38])
        in_box = (np.abs(u_xy[:, 0] - mx0) <= rx) & (np.abs(u_xy[:, 1] - my0) <= ry)
        in_ac = in_box & np.isin(cond_arr, ["MIS-C_Severe", "MIS-C_Moderate"])
        in_hd = in_box & (cond_arr == "Pediatric_Healthy")
        ax_ins.scatter(u_xy[in_hd, 0], u_xy[in_hd, 1], c="#1D4ED8", s=2.2, alpha=0.72, label="Ped.HD", rasterized=True)
        ax_ins.scatter(u_xy[in_ac, 0], u_xy[in_ac, 1], c="#B91C1C", s=2.6, alpha=0.82, label="MIS-C", rasterized=True)
        ax_ins.set_xlim(mx0 - rx, mx0 + rx)
        ax_ins.set_ylim(my0 - ry, my0 + ry)
        ax_ins.set_xticks([]); ax_ins.set_yticks([])
        full_box(ax_ins, 0.75)
        ax_ins.set_title("Myeloid ROI", fontsize=7.3, pad=1.5)
        ax_ins.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=5.8, loc="lower right", markerscale=1.3, handletextpad=0.15, borderpad=0.2)
        ind_roi = ax_a.indicate_inset_zoom(ax_ins, edgecolor=INK, alpha=0.85, lw=0.90)
        ind_roi.rectangle.set_linestyle("--")
        for c_line in ind_roi.connectors:
            c_line.set_linestyle(":")
            c_line.set_linewidth(0.75)

    handles_a = [Line2D([0], [0], marker="o", color="w", markerfacecolor=LINEAGE_COLORS[i], markeredgecolor=INK, markeredgewidth=0.55, markersize=5.0, label=LINEAGE_SHORT[i]) for i in range(9)]
    ax_a.legend(handles=handles_a, frameon=True, facecolor="white", edgecolor=INK, ncol=3, fontsize=6.8, loc="lower left")
    ax_a.set_title("18-sample surface-proteome atlas (162,777 cells)")
    ax_a.set_xlabel("Surface UMAP 1", fontsize=8.6); ax_a.set_ylabel("Surface UMAP 2", fontsize=8.6)
    plabel(ax_a, "a", x=-0.09)

    # Panel b: 3-Stage Age-Matched Clinical Density Contours (Pediatric HD -> Acute MIS-C -> Recovered)
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 5:12], wspace=0.05)
    gx, gy = atlas["gx"], atlas["gy"]
    for bi, (k_kde, b_ttl, cmap_b) in enumerate([
        ("kde_ped_hd", "Pediatric Healthy\n(n=6, 59,184 cells)", "Blues"),
        ("kde_acute", "Acute MIS-C\n(n=7, 53,999 cells)", "Reds"),
        ("kde_recov", "Paired Recovered\n(n=2, 20,635 cells)", "Greens"),
    ]):
        ax_b = fig.add_subplot(gs_b[0, bi])
        umap_corner_axes(ax_b)
        ax_b.scatter(u_xy[::4, 0], u_xy[::4, 1], c="#CBD5E1", s=0.7, linewidths=0, rasterized=True, zorder=1)
        z_k = atlas[k_kde]
        lvls = np.linspace(float(np.percentile(z_k, 56)), float(z_k.max()) * 0.96, 6)
        ax_b.contourf(gx, gy, z_k, levels=lvls, cmap=cmap_b, alpha=0.84, extend="max", zorder=2)
        ax_b.contour(gx, gy, z_k, levels=lvls, colors=INK, linewidths=0.50, zorder=3)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        full_box(ax_b, 0.85)
        ax_b.set_title(b_ttl, fontsize=9.0)
        if bi == 0:
            plabel(ax_b, "b", x=-0.14)

    # Panel c: 4-Marker Measured vs Out-of-Donor TOLT Strip on Unseen Severe MIS-C P1.1
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[1, :], wspace=0.05)
    xy_p1 = u_p1["umap_measured"]
    yt_p1, yp_p1 = u_p1["y_true"], u_p1["y_tolt"]
    pn_p1 = [str(x) for x in u_p1["protein_names"]]
    p1_mks = [("CD64", "CD64"), ("CD163", "CD163"), ("HLADR", "HLA-DR"), ("CD38", "CD38")]
    for mi, (m_raw, m_disp) in enumerate(p1_mks):
        ji = pn_p1.index(m_raw)
        vmx = max(float(np.percentile(yt_p1[:, ji], 99)), 0.8)
        r_v = float(np.corrcoef(yt_p1[:, ji], yp_p1[:, ji])[0, 1])
        for sub_j, (arr_v, tag_v) in enumerate([(yt_p1[:, ji], "Meas."), (yp_p1[:, ji], f"TOLT r={r_v:.2f}")]):
            ax_cm = fig.add_subplot(gs_c[0, mi * 2 + sub_j])
            ax_cm.scatter(xy_p1[:, 0], xy_p1[:, 1], c=arr_v, cmap=PUBCNS_SEQ, vmin=0, vmax=vmx, s=1.1, linewidths=0, rasterized=True)
            ax_cm.set_xticks([]); ax_cm.set_yticks([])
            full_box(ax_cm, 0.75)
            ax_cm.set_title(f"{m_disp} ({tag_v})", fontsize=8.0)
            if mi == 0 and sub_j == 0:
                ax_cm.set_ylabel("Unseen P1.1", fontsize=8.4, fontweight="bold")
                plabel(ax_cm, "c", x=-0.26, y=1.06)

    # Panel d: Open-Fill Donor Boxplots + All 18 Replicate Dots + Ped. HD Median Reference Line
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0:6], wspace=0.14)
    cond_ord = ["Adult_Healthy", "Pediatric_Healthy", "MIS-C_Moderate", "MIS-C_Severe", "MIS-C_Recovered"]
    cond_short = ["Ad.HD", "Ped.HD", "Mod.", "Sev.", "Recov."]
    df_p = df_path.copy()
    df_p["cond_cat"] = pd.Categorical(df_p["condition"], categories=cond_ord, ordered=True)
    for mi, (col_y, m_ttl, p_txt) in enumerate([
        ("pred_mono_CD64_mean", "Mono CD64", "** P=0.0023"),
        ("pred_mono_CD163_mean", "Mono CD163", "* P=0.0140"),
        ("pred_mono_HLADR_mean", "Mono HLA-DR", "* P=0.0221"),
    ]):
        ax_dm = fig.add_subplot(gs_d[0, mi])
        style_precision_ax(ax_dm, cat_x=True, grid="y")
        ped_med = float(df_p[df_p["condition"] == "Pediatric_Healthy"][col_y].median())
        ax_dm.axhline(ped_med, color="#1D4ED8", ls="--", lw=0.80, zorder=2)
        sns.boxplot(data=df_p, x="cond_cat", y=col_y, color="white", width=0.54, fliersize=0,
                    boxprops={"edgecolor": INK, "linewidth": 0.85, "zorder": 3}, medianprops={"color": "#B91C1C", "linewidth": 1.6},
                    whiskerprops={"color": INK, "linewidth": 0.80}, capprops={"color": INK, "linewidth": 0.80}, ax=ax_dm)
        for _, r in df_p.iterrows():
            xi = cond_ord.index(str(r["condition"])) + np.random.default_rng(abs(hash(str(r["sample_id"]))) % 10000).uniform(-0.13, 0.13)
            fc = "#B91C1C" if bool(r["has_measured_adt"]) else "#1D4ED8"
            ax_dm.scatter([xi], [float(r[col_y])], color=fc, edgecolor=INK, linewidth=0.65, s=28, zorder=4)
        yv = df_p[col_y].to_numpy(dtype=float)
        yrng = float(yv.max() - yv.min())
        ax_dm.set_ylim(float(yv.min()) - 0.10 * yrng, float(yv.max()) + 0.34 * yrng)
        sig_bracket(ax_dm, 1.0, 3.0, float(yv.max()) + 0.08 * yrng, 0.07 * yrng, p_txt, fontsize=7.4)
        ax_dm.set_xticks(range(5)); ax_dm.set_xticklabels(cond_short, rotation=28, ha="right", fontsize=7.7, fontweight="bold")
        ax_dm.set_xlabel(""); ax_dm.set_ylabel("Imputed CLR" if mi == 0 else "")
        ax_dm.set_title(m_ttl, fontsize=9.3)
        if mi == 0:
            plabel(ax_dm, "d", x=-0.24)

    # Panel e: Dual-Sided Differential Surface-Receptor Volcano with P=0.05 & BH-FDR=0.05 Threshold Lines
    ax_vol = fig.add_subplot(gs[2, 6:9])
    style_precision_ax(ax_vol, grid="both")
    dv = df_diff.copy()
    dv["neg_log10_p"] = -np.log10(np.clip(dv["p_value"].to_numpy(dtype=float), 1e-6, 1.0))
    ax_vol.axvline(0.0, color=INK, ls="-", lw=0.75, zorder=2)
    ax_vol.axvline(-0.08, color="#64748B", ls=":", lw=0.65, zorder=2)
    ax_vol.axvline(0.08, color="#64748B", ls=":", lw=0.65, zorder=2)
    ax_vol.axhline(-np.log10(0.05), color="#334155", ls=":", lw=0.80, zorder=2)
    # Approximate nominal P cutoff corresponding to BH-FDR = 0.05
    fdr_sub = dv[dv["fdr_bh"] <= 0.05]
    if len(fdr_sub):
        p_fdr_cut = float(fdr_sub["p_value"].max())
        ax_vol.axhline(-np.log10(p_fdr_cut), color="#B91C1C", ls="--", lw=0.85, zorder=2)
        ax_vol.text(-0.335, -np.log10(p_fdr_cut) - 0.14, "FDR = 0.05", fontsize=6.8, fontweight="bold", color="#B91C1C", va="top")
    up_m = (dv["delta_clr"] > 0) & (dv["p_value"] < 0.06)
    dn_m = (dv["delta_clr"] < 0) & (dv["p_value"] < 0.06)
    ns_m = ~(up_m | dn_m)
    sz_v = 22 + 38 * dv["acute_misc_mean"].to_numpy(dtype=float)
    ax_vol.scatter(dv.loc[ns_m, "delta_clr"], dv.loc[ns_m, "neg_log10_p"], s=sz_v[ns_m], color="#94A3B8", edgecolor=INK, lw=0.50, alpha=0.80, zorder=3)
    ax_vol.scatter(dv.loc[dn_m, "delta_clr"], dv.loc[dn_m, "neg_log10_p"], s=sz_v[dn_m], color="#1D4ED8", edgecolor=INK, lw=0.75, zorder=4)
    ax_vol.scatter(dv.loc[up_m, "delta_clr"], dv.loc[up_m, "neg_log10_p"], s=sz_v[up_m], color="#B91C1C", edgecolor=INK, lw=0.75, zorder=4)
    vol_callouts = {
        "Vb21.3+:CD279": (-72, 3, "#B91C1C", "Vβ21.3$^+$:PD-1"),
        "Mono:CD64": (7, 9, "#B91C1C", "Mono:CD64"),
        "Vb21.3+:CD45RO": (7, -3, "#B91C1C", "Vβ21.3$^+$:CD45RO"),
        "Mono:CD163": (7, -3, "#B91C1C", "Mono:CD163"),
        "Vb21.3+:CD38": (6, -12, "#B91C1C", "Vβ21.3$^+$:CD38"),
        "Mono:CD54": (-18, 12, "#1D4ED8", "Mono:ICAM-1"),
        "Mono:HLADR": (-22, -14, "#1D4ED8", "Mono:HLA-DR"),
    }
    for _, r in dv[dv["feature"].isin(vol_callouts)].iterrows():
        dx, dy, edge_c, lbl_c = vol_callouts[str(r["feature"])]
        ax_vol.annotate(
            lbl_c, (float(r["delta_clr"]), float(r["neg_log10_p"])),
            xytext=(dx, dy), textcoords="offset points", fontsize=7.0, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": edge_c, "lw": 0.70},
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.60}, zorder=6,
        )
    ax_vol.text(-0.335, 3.55, "Down in MIS-C", fontsize=7.2, fontweight="bold", color="#1D4ED8", va="top")
    ax_vol.text(0.50, 3.55, "Up in MIS-C", fontsize=7.2, fontweight="bold", color="#B91C1C", ha="right", va="top")
    ax_vol.set_xlim(-0.35, 0.54); ax_vol.set_ylim(-0.08, 3.78)
    ax_vol.set_xlabel("$\\Delta$CLR (Acute MIS-C $-$ Ped. HD)")
    ax_vol.set_ylabel("$-\\log_{10}(P)$")
    ax_vol.set_title("Differential receptor volcano", fontsize=9.3)
    plabel(ax_vol, "e", x=-0.16)

    # Panel f: Paired Within-Child Longitudinal Normalization (P3.1->P3.2 & P4.1->P4.2) + Ped. HD Reference
    ax_f_tr = fig.add_subplot(gs[2, 9:12])
    style_precision_ax(ax_f_tr, cat_x=True, grid="y")
    hd_sub = df_p[df_p["condition"] == "Pediatric_Healthy"]
    ax_f_tr.axhspan(float(hd_sub["pred_mono_CD64_mean"].quantile(0.25)), float(hd_sub["pred_mono_CD64_mean"].quantile(0.75)), color="#BFDBFE", alpha=0.65, label="Ped. HD CD64 IQR", zorder=1)
    ax_f_tr.axhline(float(hd_sub["pred_mono_CD64_mean"].median()), color="#1D4ED8", ls=":", lw=0.85, zorder=2)
    for pat, ac, rc, col_p, mk in [("Child P3", "P3.1", "P3.2", "#B91C1C", "o"), ("Child P4", "P4.1", "P4.2", "#1D4ED8", "s")]:
        ra = df_p[df_p["sample_id"] == ac].iloc[0]
        rr = df_p[df_p["sample_id"] == rc].iloc[0]
        ax_f_tr.plot([0, 1], [float(ra["pred_mono_CD64_mean"]), float(rr["pred_mono_CD64_mean"])], marker=mk, lw=2.1, color=col_p, markeredgecolor=INK, markeredgewidth=0.75, label=f"{pat} CD64", zorder=4)
        ax_f_tr.plot([0, 1], [float(ra["pred_mono_CD163_mean"]), float(rr["pred_mono_CD163_mean"])], marker=mk, ls="--", lw=1.55, color=col_p, markeredgecolor=INK, markeredgewidth=0.75, label=f"{pat} CD163", zorder=4)
    ax_f_tr.set_xticks([0, 1]); ax_f_tr.set_xticklabels(["Acute\n(P3.1/P4.1)", "Recovered\n(P3.2/P4.2)"], fontweight="bold")
    ax_f_tr.set_xlim(-0.22, 1.22); ax_f_tr.set_ylim(0.46, 0.96)
    ax_f_tr.set_ylabel("Monocyte surface CLR")
    ax_f_tr.set_title("Paired convalescent reversal", fontsize=9.3)
    ax_f_tr.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.5, loc="upper right")
    plabel(ax_f_tr, "f", x=-0.16)

    # Panel g: Full-Width Dendrogram + Multi-Track Metadata + 18x36 Deep Diverging Heatmap with Stars
    gs_g = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[3, :], height_ratios=[0.22, 1.0], width_ratios=[0.38, 11.6], hspace=0.02, wspace=0.01)
    ax_dend = fig.add_subplot(gs_g[0, 1])
    ax_strip = fig.add_subplot(gs_g[1, 0])
    ax_hm = fig.add_subplot(gs_g[1, 1])

    order_cond = {"Adult_Healthy": 0, "Pediatric_Healthy": 1, "MIS-C_Recovered": 2, "MIS-C_Moderate": 3, "MIS-C_Severe": 4}
    df_hm["ord"] = df_hm["condition"].map(order_cond)
    df_hm = df_hm.sort_values(["ord", "sample_id"]).reset_index(drop=True)
    f_cols = [c for c in df_hm.columns if ":" in c][:36]
    mat_raw = df_hm[f_cols].to_numpy(dtype=float)
    mat_z = (mat_raw - np.nanmean(mat_raw, axis=0, keepdims=True)) / np.maximum(np.nanstd(mat_raw, axis=0, keepdims=True), 1e-6)
    col_link = linkage(pdist(mat_z.T, metric="euclidean"), method="ward")
    dend = dendrogram(col_link, ax=ax_dend, color_threshold=0, above_threshold_color=INK)
    ax_dend.set_xticks([]); ax_dend.set_yticks([])
    for sp_s in ["top", "right", "left", "bottom"]:
        ax_dend.spines[sp_s].set_visible(False)
    ax_dend.set_title("Cohort-wide 18-sample × 36-compartment surface immunophenotype matrix (carmine frames & stars: differential MIS-C modules)", fontsize=9.3, pad=2)

    col_ord = dend["leaves"]
    f_cols_ord = [f_cols[i] for i in col_ord]
    mat_z_ord = np.clip(mat_z[:, col_ord], -2.3, 2.3)

    cond_cmap = {"Adult_Healthy": "#475569", "Pediatric_Healthy": "#1D4ED8", "MIS-C_Recovered": "#047857", "MIS-C_Moderate": "#B45309", "MIS-C_Severe": "#B91C1C"}
    for ri, r_meta in df_hm.iterrows():
        ax_strip.add_patch(mpatches.Rectangle((0, ri), 0.55, 1, facecolor=cond_cmap[str(r_meta["condition"])], edgecolor=INK, lw=0.50))
        ax_strip.add_patch(mpatches.Rectangle((0.55, ri), 0.45, 1, facecolor="#5B21B6" if bool(r_meta["has_adt"]) else "#E2E8F0", edgecolor=INK, lw=0.50))
    ax_strip.set_xlim(0, 1); ax_strip.set_ylim(len(df_hm), 0)
    full_box(ax_strip, 0.80)
    ax_strip.tick_params(length=0)
    ax_strip.set_xticks([0.27, 0.77]); ax_strip.set_xticklabels(["Stage", "ADT"], rotation=90, fontsize=7.0, fontweight="bold")
    ax_strip.set_yticks(np.arange(len(df_hm)) + 0.5)
    ax_strip.set_yticklabels(df_hm["sample_id"].tolist(), rotation=0, fontsize=7.7, fontweight="bold")
    plabel(ax_strip, "g", x=-1.15, y=1.16)

    p_lookup = dict(zip(df_diff["feature"], df_diff["p_value"]))
    sev_start = int(np.flatnonzero(df_hm["condition"] == "MIS-C_Severe")[0])
    disp_cols = [c.replace("HLADR", "HLA-DR").replace("CD279", "PD-1") for c in f_cols_ord]
    sns.heatmap(
        mat_z_ord, annot=False, cmap=PUBCNS_DIV, vmin=-2.2, vmax=2.2,
        xticklabels=disp_cols, yticklabels=False,
        linewidths=0.45, linecolor=INK, cbar_kws={"label": "Donor Z-score", "pad": 0.012, "fraction": 0.018},
        ax=ax_hm,
    )
    for ci, cname in enumerate(f_cols_ord):
        pv = float(p_lookup.get(cname, 1.0))
        st = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        if st:
            zv = mat_z_ord[sev_start, ci]
            tc = "white" if abs(zv) > 1.05 else INK
            ax_hm.text(ci + 0.5, sev_start + 0.55, st, ha="center", va="center", fontsize=9.2, fontweight="bold", color=tc, zorder=5)
    full_box(ax_hm, 0.85)
    ax_hm.tick_params(length=0)
    ax_hm.set_xticklabels(disp_cols, rotation=46, ha="right", fontsize=7.7, fontweight="bold")
    key_hi = {"Mono:CD64", "Mono:CD163", "Mono:HLADR", "Mono:CD54", "Vb21.3+:CD38", "Vb21.3+:CD279", "Vb21.3+:TIGIT", "Vb21.3+:CD45RO"}
    for ci, cname in enumerate(f_cols_ord):
        if cname in key_hi:
            ax_hm.add_patch(mpatches.Rectangle((ci, 0), 1, len(df_hm), fill=False, edgecolor="#B91C1C", lw=1.65, clip_on=False, zorder=6))

    save_fig(fig, MAIN_FIG_DIR, "Figure5")


# =============================================================================
# FIGURE 6: Single-Cell TCR TRBV11-2 Superantigen Clones & Uncertainty (a-g)
# =============================================================================
def make_figure6_tcr_superantigen_and_uncertainty() -> None:
    df_vb = pd.read_csv(TABLES_DIR / "v41_tcr_vbeta_landscape.csv")
    df_tcr = pd.read_csv(TABLES_DIR / "gse166489_tcr_trbv11_2_phenotyping.csv")
    df_unc = pd.read_csv(TABLES_DIR / "v41_uncertainty_calibration.csv")
    tcr_sc = np.load(FIGURE_PRED_DIR / "v41_tcr_singlecell_distributions.npz", allow_pickle=True)
    atlas = np.load(FIGURE_PRED_DIR / "v41_misc_18donor_atlas.npz", allow_pickle=True)

    fig = plt.figure(figsize=(12.6, 8.14))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[0.96, 1.04, 0.96], hspace=0.33, wspace=0.88)

    # Panel a: 18-Sample x Top-12 TRBV Family Repertoire Heatmap
    ax_a = fig.add_subplot(gs[0, 0:8])
    top_v = df_vb.groupby("v_gene")["pct"].max().sort_values(ascending=False).index[:15].tolist()
    if "TRBV11-2" in top_v:
        top_v = ["TRBV11-2"] + [v for v in top_v if v != "TRBV11-2"]
    s_ord = ["A.HD1", "A.HD2", "A.HD3", "C.HD1", "C.HD2", "C.HD3", "C.HD4", "C.HD5", "C.HD6", "P3.2", "P4.2", "P5.1", "P4.1", "P6.1", "P7.1", "P2.1", "P1.1", "P3.1"]
    piv_v = df_vb[df_vb["v_gene"].isin(top_v)].pivot(index="v_gene", columns="sample_id", values="pct").fillna(0.0)
    piv_v = piv_v.loc[[v for v in top_v[:12] if v in piv_v.index], [s for s in s_ord if s in piv_v.columns]]
    sns.heatmap(piv_v, cmap=PUBCNS_SEQ, vmin=0, vmax=24, linewidths=0.55, linecolor=INK,
                cbar_kws={"label": "% of TCR$^+$ T cells", "pad": 0.015}, ax=ax_a)
    full_box(ax_a, 0.85)
    ax_a.tick_params(length=0)
    ax_a.add_patch(mpatches.Rectangle((0, 0), piv_v.shape[1], 1, fill=False, edgecolor="#B91C1C", lw=1.9, zorder=6))
    ax_a.set_xticklabels(piv_v.columns, rotation=32, ha="right", fontsize=8.0, fontweight="bold")
    ax_a.set_yticklabels(piv_v.index, rotation=0, fontsize=8.0, fontweight="bold")
    ax_a.set_xlabel(""); ax_a.set_ylabel("")
    ax_a.set_title("Productive $\\alpha\\beta$ TCR V$\\beta$ repertoire across all 18 donors isolates polyclonal TRBV11-2 expansion")
    plabel(ax_a, "a", x=-0.09)

    # Panel b: Integrated T-Cell Manifold Highlighting TRBV11-2+ Clones Across CD4+ & CD8+ Islands
    ax_b = fig.add_subplot(gs[0, 8:12])
    umap_corner_axes(ax_b)
    u_xy = atlas["umap_xy"]
    lin = atlas["lineages"]
    is_v21 = atlas["is_trbv11_2"].astype(bool)
    conds = atlas["conditions"]
    t_idx = np.isin(lin, [0, 1, 2, 3])
    v21_cd4 = np.isin(lin, [0, 1]) & is_v21 & np.isin(conds, ["MIS-C_Severe", "MIS-C_Moderate"])
    v21_cd8 = np.isin(lin, [2, 3]) & is_v21 & np.isin(conds, ["MIS-C_Severe", "MIS-C_Moderate"])
    ax_b.scatter(u_xy[t_idx, 0], u_xy[t_idx, 1], c="#CBD5E1", s=1.4, linewidths=0, label="Bystander T", rasterized=True, zorder=1)
    ax_b.scatter(u_xy[v21_cd4, 0], u_xy[v21_cd4, 1], c="#B91C1C", edgecolor=INK, lw=0.40, s=16, label="CD4$^+$ Vβ21.3$^+$", zorder=4)
    ax_b.scatter(u_xy[v21_cd8, 0], u_xy[v21_cd8, 1], c="#5B21B6", edgecolor=INK, lw=0.40, s=17, marker="^", label="CD8$^+$ Vβ21.3$^+$", zorder=5)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    full_box(ax_b, 0.85)
    ax_b.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.1, loc="lower left")
    ax_b.set_title("Polyclonal TRBV11-2$^+$ T-cell niche")
    ax_b.set_xlabel("UMAP 1", fontsize=8.6); ax_b.set_ylabel("UMAP 2", fontsize=8.6)
    plabel(ax_b, "b", x=-0.13)

    # Panel c: Raw mRNA Zero-UMI Dropout Floor vs TOLT Surface Protein Resolution (PDCD1/PD-1 & CD38)
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0:6], wspace=0.22)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    style_precision_ax(ax_c1, cat_x=True, grid="y")
    ax_c1.axhline(0.0, color="#B91C1C", ls="--", lw=0.85, zorder=2)
    df_v_mrna = pd.DataFrame({
        "val": np.concatenate([tcr_sc["pd1_mrna_neg"], tcr_sc["pd1_mrna_pos"], tcr_sc["cd38_mrna_neg"], tcr_sc["cd38_mrna_pos"]]),
        "target": ["PDCD1\nmRNA"] * (len(tcr_sc["pd1_mrna_neg"]) + len(tcr_sc["pd1_mrna_pos"])) + ["CD38\nmRNA"] * (len(tcr_sc["cd38_mrna_neg"]) + len(tcr_sc["cd38_mrna_pos"])),
        "clone": (["Non-Vβ21.3"] * len(tcr_sc["pd1_mrna_neg"]) + ["TRBV11-2+"] * len(tcr_sc["pd1_mrna_pos"])) + (["Non-Vβ21.3"] * len(tcr_sc["cd38_mrna_neg"]) + ["TRBV11-2+"] * len(tcr_sc["cd38_mrna_pos"])),
    })
    sns.violinplot(data=df_v_mrna, x="target", y="val", hue="clone", split=True, cut=0, inner="quart",
                   palette={"Non-Vβ21.3": "#64748B", "TRBV11-2+": "#B91C1C"}, linewidth=0.85, ax=ax_c1)
    z_pd1 = 100.0 * float((tcr_sc["pd1_mrna_pos"] == 0).mean())
    z_38 = 100.0 * float((tcr_sc["cd38_mrna_pos"] == 0).mean())
    ax_c1.text(0.5, 0.94, f"0-UMI dropout:\nPDCD1 {z_pd1:.1f}%\nCD38 {z_38:.1f}%", transform=ax_c1.transAxes, ha="center", va="top",
               fontsize=7.3, fontweight="bold", bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.65})
    ax_c1.set_xlabel(""); ax_c1.set_ylabel("Raw mRNA log1p(CP10k)")
    ax_c1.set_title("Raw scRNA-seq zero floor", fontsize=9.3)
    ax_c1.get_legend().remove()
    plabel(ax_c1, "c", x=-0.18)

    ax_c2 = fig.add_subplot(gs_c[0, 1])
    style_precision_ax(ax_c2, cat_x=True, grid="y")
    ax_c2.axhline(1.05, color="#B91C1C", ls=":", lw=0.80, zorder=2)
    df_v_adt = pd.DataFrame({
        "val": np.concatenate([tcr_sc["pd1_adt_neg"], tcr_sc["pd1_adt_pos"], tcr_sc["cd38_adt_neg"], tcr_sc["cd38_adt_pos"]]),
        "target": ["Surface\nPD-1"] * (len(tcr_sc["pd1_adt_neg"]) + len(tcr_sc["pd1_adt_pos"])) + ["Surface\nCD38"] * (len(tcr_sc["cd38_adt_neg"]) + len(tcr_sc["cd38_adt_pos"])),
        "clone": (["Non-Vβ21.3"] * len(tcr_sc["pd1_adt_neg"]) + ["TRBV11-2+"] * len(tcr_sc["pd1_adt_pos"])) + (["Non-Vβ21.3"] * len(tcr_sc["cd38_adt_neg"]) + ["TRBV11-2+"] * len(tcr_sc["cd38_adt_pos"])),
    })
    sns.violinplot(data=df_v_adt, x="target", y="val", hue="clone", split=True, cut=0, inner="quart",
                   palette={"Non-Vβ21.3": "#64748B", "TRBV11-2+": "#B91C1C"}, linewidth=0.85, ax=ax_c2)
    ax_c2.set_xlabel(""); ax_c2.set_ylabel("TOLT surface CLR")
    ax_c2.set_title("TOLT surface resolution", fontsize=9.3)
    ax_c2.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.9, loc="upper right")

    # Panel d: Within-Patient Autologous Paired Slopegraphs + Negative Control Patient P6.1
    ax_d = fig.add_subplot(gs[1, 6:12])
    style_precision_ax(ax_d, cat_x=True, grid="y")
    mks_d = [("pred_CD38_mean", "CD38"), ("pred_HLADR_mean", "HLA-DR"), ("pred_PD1_mean", "PD-1"), ("pred_TIGIT_mean", "TIGIT"), ("pred_CD45RO_mean", "CD45RO")]
    for mi, (col_m, lbl_m) in enumerate(mks_d):
        if mi % 2 == 1:
            ax_d.axvspan(mi - 0.45, mi + 0.45, color="#F8FAFC", zorder=0)
        if mi > 0:
            ax_d.axvline(mi - 0.5, color="#94A3B8", ls="--", lw=0.60, zorder=1)
        for sid in ["P1.1", "P2.1", "P3.1", "P4.1", "P7.1"]:
            rn = df_tcr[(df_tcr["sample_id"] == sid) & (df_tcr["clone_group"] == "TRBV11-2_neg")]
            rp = df_tcr[(df_tcr["sample_id"] == sid) & (df_tcr["clone_group"] == "TRBV11-2_pos")]
            if len(rn) and len(rp):
                vn, vp = float(rn[col_m].iloc[0]), float(rp[col_m].iloc[0])
                ax_d.plot([mi - 0.18, mi + 0.18], [vn, vp], color="#B91C1C", lw=1.35, alpha=0.92, zorder=3)
                ax_d.scatter([mi - 0.18], [vn], color="#1D4ED8", edgecolor=INK, lw=0.65, s=26, zorder=4)
                ax_d.scatter([mi + 0.18], [vp], color="#B91C1C", edgecolor=INK, lw=0.65, s=30, zorder=4)
        rn6 = df_tcr[(df_tcr["sample_id"] == "P6.1") & (df_tcr["clone_group"] == "TRBV11-2_neg")]
        rp6 = df_tcr[(df_tcr["sample_id"] == "P6.1") & (df_tcr["clone_group"] == "TRBV11-2_pos")]
        if len(rn6) and len(rp6):
            ax_d.plot([mi - 0.18, mi + 0.18], [float(rn6[col_m].iloc[0]), float(rp6[col_m].iloc[0])], color="#047857", ls="--", lw=1.7, zorder=5)
    h_d = [
        Line2D([0], [0], color="#B91C1C", marker="o", markeredgecolor=INK, lw=1.45, label="Expanded MIS-C (n=5: Vβ21.3$^-$ $\\to$ Vβ21.3$^+$)"),
        Line2D([0], [0], color="#047857", ls="--", lw=1.7, label="Non-expanded MIS-C P6.1 (specificity control)"),
    ]
    ax_d.set_xticks(range(len(mks_d))); ax_d.set_xticklabels([m[1] for m in mks_d], fontweight="bold")
    ax_d.set_ylim(0.35, 1.48)
    ax_d.set_ylabel("Autologous donor T-cell CLR")
    ax_d.set_title("Clonotype-intrinsic co-activation vs autologous bystanders")
    ax_d.legend(handles=h_d, frameon=True, facecolor="white", edgecolor=INK, fontsize=7.2, loc="upper left")
    plabel(ax_d, "d", x=-0.09)

    # Panel e: In Silico 2D Flow-Cytometric Density Contour + Precision Quadrant Gate
    ax_e = fig.add_subplot(gs[2, 0:4])
    style_precision_ax(ax_e, grid="both")
    x_neg, y_neg = tcr_sc["cd38_adt_neg"], tcr_sc["pd1_adt_neg"]
    x_pos, y_pos = tcr_sc["cd38_adt_pos"], tcr_sc["pd1_adt_pos"]
    is_cd4_p = tcr_sc["cd4_adt_pos"] >= tcr_sc["cd8_adt_pos"]
    if len(x_neg) > 100:
        sub_n = np.random.default_rng(7).choice(len(x_neg), size=min(1500, len(x_neg)), replace=False)
        kde_n = gaussian_kde(np.vstack([x_neg[sub_n], y_neg[sub_n]]))
        xg, yg_c = np.meshgrid(np.linspace(0.0, 1.6, 50), np.linspace(0.2, 1.9, 50))
        zg = kde_n(np.vstack([xg.ravel(), yg_c.ravel()])).reshape(xg.shape)
        ax_e.contour(xg, yg_c, zg, levels=5, colors="#475569", linewidths=0.75, alpha=0.90, zorder=2)
    ax_e.scatter(x_neg[::6], y_neg[::6], c="#94A3B8", s=3.5, alpha=0.50, label="Bystander T", zorder=1)
    ax_e.scatter(x_pos[is_cd4_p], y_pos[is_cd4_p], c="#B91C1C", edgecolor=INK, lw=0.45, s=16, alpha=0.90, label="CD4$^+$ Vβ21.3$^+$", zorder=4)
    ax_e.scatter(x_pos[~is_cd4_p], y_pos[~is_cd4_p], c="#5B21B6", edgecolor=INK, lw=0.45, s=17, alpha=0.90, marker="^", label="CD8$^+$ Vβ21.3$^+$", zorder=5)
    ax_e.axvline(0.75, color=INK, ls="--", lw=0.95, zorder=5); ax_e.axhline(1.05, color=INK, ls="--", lw=0.95, zorder=5)
    dp_pos = 100.0 * float(((x_pos >= 0.75) & (y_pos >= 1.05)).mean())
    dp_neg = 100.0 * float(((x_neg >= 0.75) & (y_neg >= 1.05)).mean())
    ax_e.text(0.03, 0.95, "Q1", transform=ax_e.transAxes, fontsize=7.3, fontweight="bold", color="#475569", va="top")
    ax_e.text(0.97, 0.04, "Q4", transform=ax_e.transAxes, fontsize=7.3, fontweight="bold", color="#475569", ha="right")
    ax_e.text(0.97, 0.95, f"Q2 CD38$^{{\\mathrm{{hi}}}}$PD-1$^+$:\nVβ21.3$^+$: {dp_pos:.1f}%\nBystander: {dp_neg:.1f}%",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=7.4, fontweight="bold",
              bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.70}, zorder=6)
    ax_e.set_xlim(0.05, 1.65); ax_e.set_ylim(0.25, 1.92)
    ax_e.set_xlabel("Imputed surface CD38 (CLR)")
    ax_e.set_ylabel("Imputed surface PD-1 (CLR)")
    ax_e.set_title("In silico 2D flow cytometry gate")
    ax_e.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.0, loc="lower left")
    plabel(ax_e, "e", x=-0.14)

    # Panels f & g: Per-Cell Heteroscedastic Uncertainty Calibration + Cohort Mean & OLS Reference Lines
    for ax_u, coh_u, ttl_u, ltr_u, col_u in [
        (fig.add_subplot(gs[2, 4:8]), "GSE194122_Unseen_Site4", "Uncertainty calibration (unseen Site 4 BMMC)", "f", "#1D4ED8"),
        (fig.add_subplot(gs[2, 8:12]), "GSE166489_Unseen_MISC_P1.1", "Uncertainty calibration (unseen MIS-C P1.1)", "g", "#B91C1C"),
    ]:
        style_precision_ax(ax_u, grid="both")
        su = df_unc[df_unc["cohort"] == coh_u].sort_values("decile")
        xd_u = su["decile"].to_numpy(dtype=float)
        yv_u = su["mean_cell_rmse"].to_numpy(dtype=float)
        ax_u.axhline(float(yv_u.mean()), color="#475569", ls=":", lw=0.80, zorder=2)
        sl_u, ic_u = np.polyfit(xd_u, yv_u, 1)
        ax_u.plot(xd_u, sl_u * xd_u + ic_u, color=INK, ls="--", lw=0.95, zorder=4)
        ax_u.bar(xd_u, yv_u, width=0.66, color=col_u, alpha=0.90, edgecolor=INK, linewidth=0.80, zorder=3)
        ax_u.plot(xd_u, yv_u, color=INK, marker="o", markerfacecolor="white", markeredgewidth=0.90, markersize=4.5, lw=1.35, zorder=5)
        rho_u, p_u = spearmanr(xd_u, yv_u)
        ax_u.text(0.05, 0.93, f"Spearman $\\rho = {rho_u:.2f}$\n($P = {p_u:.1e}$)", transform=ax_u.transAxes, va="top",
                  fontsize=7.8, fontweight="bold", bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": INK, "lw": 0.70}, zorder=6)
        ax_u.set_xticks(range(1, 11))
        ax_u.set_xlabel("Predicted uncertainty decile (low $\\to$ high $\\hat{\\sigma}_i$)")
        ax_u.set_ylabel("Empirical per-cell RMSE")
        ymin_u, ymax_u = float(yv_u.min()), float(yv_u.max())
        ax_u.set_ylim(max(0.0, ymin_u - 0.25 * (ymax_u - ymin_u)), ymax_u + 0.26 * (ymax_u - ymin_u))
        ax_u.set_title(ttl_u, fontsize=9.3)
        plabel(ax_u, ltr_u, x=-0.14)

    save_fig(fig, MAIN_FIG_DIR, "Figure6")


# =============================================================================
# FIGURE 7: 5-Well Cross-Donor TEA-seq, Genome-Browser Chromatin & 9-Lineage Flow (a-h)
# =============================================================================
def make_figure7_teaseq_triomic_and_chromatin() -> None:
    df_sum = pd.read_csv(TABLES_DIR / "v41_teaseq_complete_summary.csv")
    df_gain = pd.read_csv(TABLES_DIR / "teaseq_per_antibody_chromatin_gain.csv")
    df_trk = pd.read_csv(TABLES_DIR / "v41_teaseq_cis_genomic_tracks.csv")
    df_zm = pd.read_csv(TABLES_DIR / "v41_teaseq_zero_mrna_rescue.csv")
    df_cm = pd.read_csv(TABLES_DIR / "v41_teaseq_confusion_matrix.csv")
    u_x1 = np.load(FIGURE_PRED_DIR / "teaseq_umap_X066_W3_to_X061_W1_full.npz", allow_pickle=True)

    fig = plt.figure(figsize=(12.95, 8.67))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[0.96, 1.08, 1.02], hspace=0.34, wspace=0.85)

    # Panel a: Complete Multi-Well & Cross-Donor TEA-seq Benchmark
    ax_a = fig.add_subplot(gs[0, 0:7])
    style_precision_ax(ax_a, cat_x=True, grid="y")
    tiers_a = [
        ("Within_Well_scVAEIT_Reference", "full_3534", "Within-Well\n(X066 W3)"),
        ("Cross_Well_Same_Donor_HMN200910", "250", "Cross-Well\n(n=250)"),
        ("Cross_Well_Same_Donor_HMN200910", "full", "Cross-Well\n(Full)"),
        ("True_Cross_Donor_Triomic", "250", "Cross-Donor\n(n=250)"),
        ("True_Cross_Donor_Triomic", "full", "Cross-Donor\n(Full)"),
    ]
    m_a = ["scVAEIT_RNA+ATAC", "Classical_RNA+ATAC", "scLinear_RNA", "TOLT_RNA", "TOLT_RNA+ATAC"]
    m_lbl_a = {"scVAEIT_RNA+ATAC": "scVAEIT (RNA+ATAC)", "Classical_RNA+ATAC": "SVD-Ridge (RNA+ATAC)", "scLinear_RNA": "scLinear (RNA)", "TOLT_RNA": "TOLT (RNA)", "TOLT_RNA+ATAC": "TOLT (RNA+ATAC)"}
    xa = np.arange(len(tiers_a))
    wa = 0.158
    ax_a.axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    for ti in range(len(tiers_a)):
        if ti % 2 == 1:
            ax_a.axvspan(ti - 0.46, ti + 0.46, color="#F8FAFC", zorder=0)
        if ti > 0:
            ax_a.axvline(ti - 0.5, color="#94A3B8", ls="--", lw=0.65)
    for mi, m in enumerate(m_a):
        vals = []
        for ti, (t_tier, b_code, _) in enumerate(tiers_a):
            s_v = df_sum[(df_sum["transfer_tier"] == t_tier) & (df_sum["budget"].astype(str) == b_code) & (df_sum["method"] == m)]["clean45_pcc_median"].to_numpy(dtype=float)
            vals.append(float(s_v.mean()) if len(s_v) else 0.0)
            if len(s_v) > 1:
                jit = np.linspace(-0.025, 0.025, len(s_v))
                ax_a.scatter(ti + (mi - 2) * wa + jit, s_v, s=12, facecolor="white", edgecolor=INK, lw=0.60, zorder=5)
        ax_a.bar(xa + (mi - 2) * wa, vals, width=wa, color=PALETTE[m], edgecolor=INK, linewidth=0.75, label=m_lbl_a[m], zorder=3)
    sig_bracket(ax_a, 3 - 2 * wa, 3 + 2 * wa, 0.57, 0.02, "+0.18 vs scVAEIT", fontsize=7.4)
    ax_a.set_xticks(xa); ax_a.set_xticklabels([t[2] for t in tiers_a], fontweight="bold")
    ax_a.set_ylim(0.20, 0.735)
    ax_a.set_ylabel("45-antibody median PCC")
    ax_a.set_title("5-well & cross-donor trimodal TEA-seq translation (GSE158013)")
    ax_a.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=3, fontsize=7.1, loc="upper left")
    plabel(ax_a, "a", x=-0.09)

    # Panel b: All 45 Human Antibodies Diagonal Chromatin Synergy on Unseen Donor X061_W1
    ax_b = fig.add_subplot(gs[0, 7:12])
    style_precision_ax(ax_b, grid="both")
    cd_g = df_gain[(df_gain["pair"] == "X066_W3_to_X061_W1") & (df_gain["budget"] == "full")].copy()
    improved = cd_g["delta_atac_tolt"] > 0
    ax_b.plot([0.05, 0.95], [0.05, 0.95], color=INK, ls="--", lw=0.90, zorder=2)
    ax_b.plot([0.05, 0.90], [0.10, 0.95], color="#B91C1C", ls=":", lw=0.75, zorder=2)
    ax_b.scatter(cd_g.loc[~improved, "tolt_rna_pcc"], cd_g.loc[~improved, "tolt_rna_atac_pcc"], color="#64748B", edgecolor=INK, lw=0.65, s=38, label=f"RNA ≥ Dual ({(~improved).sum()}/45)", zorder=3)
    ax_b.scatter(cd_g.loc[improved, "tolt_rna_pcc"], cd_g.loc[improved, "tolt_rna_atac_pcc"], color="#B91C1C", edgecolor=INK, lw=0.80, s=50, label=f"RNA+ATAC > RNA ({improved.sum()}/45, P=9.7e-6)", zorder=4)
    off_7b = {"CD80": (12, -2), "TCR-Va7.2": (-44, 5), "CD40": (-28, 5), "CD8a": (-40, -9), "CD172a": (-62, 5), "CD11c": (-18, -15)}
    for _, r in cd_g.nlargest(6, "delta_atac_tolt").iterrows():
        ab_nm = str(r["antibody"])
        ax_b.annotate(
            ab_nm, (float(r["tolt_rna_pcc"]), float(r["tolt_rna_atac_pcc"])),
            xytext=off_7b.get(ab_nm, (-28, 4)), textcoords="offset points", fontsize=7.3, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": INK, "lw": 0.55},
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.55}, zorder=5,
        )
    ax_b.set_xlim(0.05, 0.95); ax_b.set_ylim(0.05, 0.95)
    ax_b.set_xlabel("TOLT (RNA only) unseen-donor PCC")
    ax_b.set_ylabel("TOLT (RNA + ATAC) unseen-donor PCC")
    ax_b.set_title("Unseen-donor chromatin synergy (X066→X061)")
    ax_b.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.1, loc="lower right")
    plabel(ax_b, "b", x=-0.12)

    # Panel c: Genome-Browser Smooth Chromatin Profiles + RefSeq Gene Structure at ITGAX (CD11c) Locus
    gs_c = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[1, 0:6], height_ratios=[1, 1, 1, 1, 0.85, 0.58], hspace=0.06)
    itg = df_trk[df_trk["locus"] == "ITGAX (CD11c)"].sort_values("mid_kb")
    lin_tracks = [
        ("open_pct_Myeloid_Mono_cDC", "Myeloid", "#B45309"),
        ("open_pct_B_Mem_CD27", "B Memory", "#5B21B6"),
        ("open_pct_NK_CD56_CD16", "NK cells", "#B91C1C"),
        ("open_pct_CD4_Naive_RA", "CD4 Naive", "#1D4ED8"),
    ]
    xmin_k, xmax_k = 31347.0, 31376.0
    tss_k = 31355.13
    xg_tr = np.linspace(xmin_k, xmax_k, 400)
    for ti, (col_tr, lbl_tr, col_f) in enumerate(lin_tracks):
        ax_tr = fig.add_subplot(gs_c[ti, 0])
        full_box(ax_tr, 0.75)
        ax_tr.set_facecolor("#F8FAFC" if ti == 0 else "white")
        prof = np.zeros_like(xg_tr)
        for _, pr in itg.iterrows():
            sig_w = max((float(pr["end"]) - float(pr["start"])) / 2200.0, 0.42)
            prof += float(pr[col_tr]) * np.exp(-0.5 * ((xg_tr - float(pr["mid_kb"])) / sig_w) ** 2)
        ax_tr.axhline(25.0, color="#E2E8F0", ls=":", lw=0.45, zorder=1)
        ax_tr.fill_between(xg_tr, 0, prof, color=col_f, alpha=0.88, zorder=2)
        ax_tr.plot(xg_tr, prof, color=INK, lw=0.70, zorder=3)
        ax_tr.axvline(tss_k, color="#B91C1C", ls="--", lw=0.90, zorder=4)
        ax_tr.set_xlim(xmin_k, xmax_k); ax_tr.set_ylim(0, 62)
        ax_tr.set_xticks([]); ax_tr.set_yticks([50])
        ax_tr.set_ylabel(lbl_tr, fontsize=7.5, fontweight="bold", rotation=0, labelpad=24, va="center")
        if ti == 0:
            ax_tr.set_title("GRCh38 cis-chromatin tracks at ITGAX (CD11c) locus (chr16)", fontsize=9.1)
            plabel(ax_tr, "c", x=-0.13, y=1.20)

    ax_pk = fig.add_subplot(gs_c[4, 0])
    full_box(ax_pk, 0.75)
    for _, pr in itg.iterrows():
        w_kb = max((float(pr["end"]) - float(pr["start"])) / 1000.0, 0.9)
        ax_pk.bar(float(pr["mid_kb"]), max(0.0, float(pr["peak_adt_pcc"])), width=w_kb, color="#5B21B6", edgecolor=INK, linewidth=0.65)
    ax_pk.axvline(tss_k, color="#B91C1C", ls="--", lw=0.90)
    ax_pk.set_xlim(xmin_k, xmax_k); ax_pk.set_ylim(0, 0.62)
    ax_pk.set_xticks([]); ax_pk.set_yticks([0.5])
    ax_pk.set_ylabel("Peak $r$", fontsize=7.5, fontweight="bold", rotation=0, labelpad=24, va="center")

    ax_gn = fig.add_subplot(gs_c[5, 0])
    for sp_g in ["top", "right", "left"]:
        ax_gn.spines[sp_g].set_visible(False)
    ax_gn.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_gn.tick_params(axis="x", which="both", top=False, direction="out")
    ax_gn.plot([tss_k, 31375.2], [0.5, 0.5], color=INK, lw=1.05)
    for ex_x in [31355.13, 31356.8, 31359.4, 31362.1, 31365.0, 31368.5, 31371.8, 31374.4]:
        ax_gn.add_patch(mpatches.Rectangle((ex_x, 0.18), 0.55, 0.64, facecolor=INK, edgecolor=INK, lw=0.4))
    ax_gn.annotate("", xy=(tss_k + 2.2, 0.88), xytext=(tss_k, 0.88), arrowprops={"arrowstyle": "->", "color": "#B91C1C", "lw": 1.05})
    ax_gn.text(tss_k - 0.5, 0.5, "ITGAX", ha="right", va="center", fontsize=7.9, fontweight="bold", style="italic", color=INK)
    ax_gn.set_xlim(xmin_k, xmax_k); ax_gn.set_ylim(0, 1.05)
    ax_gn.set_yticks([])
    ax_gn.set_xlabel("Genomic coordinate on chr16 (kb, GRCh38)", fontsize=8.7)

    # Panel d: Zero-mRNA Dropout Chromatin Rescue on Unseen Donor X061_W1
    ax_d = fig.add_subplot(gs[1, 7:12])
    style_precision_ax(ax_d, cat_y=True, grid="x")
    zm_ord = df_zm.sort_values("delta_chromatin_rescue", ascending=True).tail(8)
    yd = np.arange(len(zm_ord))
    ax_d.axvline(0.50, color="#64748B", ls=":", lw=0.80, zorder=2)
    ax_d.barh(yd - 0.18, zm_ord["tolt_rna_pcc_in_zero_mrna"], height=0.35, color="#1D4ED8", edgecolor=INK, lw=0.75, label="TOLT (RNA) in 0-UMI cells", zorder=3)
    ax_d.barh(yd + 0.18, zm_ord["tolt_rna_atac_pcc_in_zero_mrna"], height=0.35, color="#B91C1C", edgecolor=INK, lw=0.75, label="TOLT (RNA+ATAC) in 0-UMI cells", zorder=3)
    lbls_zm = [f"{r['antibody']} ({r['zero_mrna_pct']:.0f}% 0-mRNA)" for _, r in zm_ord.iterrows()]
    ax_d.set_yticks(yd); ax_d.set_yticklabels(lbls_zm, fontsize=8.0, fontweight="bold")
    ax_d.set_xlabel("Surface protein PCC within cognate mRNA = 0 cells")
    ax_d.set_title("Open chromatin rescues mRNA-dropout cells")
    ax_d.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.3, loc="lower right")
    plabel(ax_d, "d", x=-0.20)

    # Panels e & f: Unseen Donor X061_W1 Measured vs TOLT RNA+ATAC 9-Lineage Flow UMAPs
    gs_ef = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0:4], wspace=0.08)
    xy_x1 = u_x1["umap_measured"]
    for sub_i, (l_key, uttl, ltr_u) in enumerate([
        ("true_9lineage", "Measured 45-ADT gates", "e"),
        ("pred_9lineage_p2", "TOLT RNA+ATAC gates", "f"),
    ]):
        ax_u = fig.add_subplot(gs_ef[0, sub_i])
        umap_corner_axes(ax_u)
        cls_v = u_x1[l_key]
        ax_u.scatter(xy_x1[:, 0], xy_x1[:, 1], c=[LINEAGE_COLORS[int(c)] for c in cls_v], s=1.6, alpha=0.90, linewidths=0, rasterized=True, zorder=2)
        ax_u.set_xticks([]); ax_u.set_yticks([])
        full_box(ax_u, 0.85)
        ax_u.set_title(uttl, fontsize=9.3)
        ax_u.set_xlabel("UMAP 1 (X061_W1)", fontsize=8.4)
        if sub_i == 0:
            ax_u.set_ylabel("UMAP 2", fontsize=8.4)
        plabel(ax_u, ltr_u, x=-0.15)

    # Panel g: 9x9 PBMC Flow-Gating Normalized Confusion Matrix (High-Contrast Prussian Navy Colormap)
    ax_g = fig.add_subplot(gs[2, 5:9])
    cm_w3 = df_cm[df_cm["benchmark"] == "Within_Well_X066_W3"].pivot(index="true_lineage", columns="pred_lineage", values="row_recall")
    from triomics.v4.immunophenotyping import NINE_LINEAGE_CLASSES
    ord_l = [l for l in NINE_LINEAGE_CLASSES if l in cm_w3.index]
    mat_cm = cm_w3.loc[ord_l, ord_l].to_numpy(dtype=float)
    deep_blues = LinearSegmentedColormap.from_list("DeepBlues", ["#F8FAFC", "#BFDBFE", "#3B82F6", "#1D4ED8", "#0B3C5D"])
    sns.heatmap(mat_cm, annot=False, cmap=deep_blues, vmin=0.0, vmax=0.95, cbar=False,
                xticklabels=LINEAGE_SHORT[:len(ord_l)], yticklabels=LINEAGE_SHORT[:len(ord_l)],
                linewidths=0.60, linecolor=INK, square=True, ax=ax_g)
    for ri in range(mat_cm.shape[0]):
        for ci in range(mat_cm.shape[1]):
            tc = "white" if mat_cm[ri, ci] >= 0.45 else INK
            ax_g.text(ci + 0.5, ri + 0.5, f"{mat_cm[ri, ci]:.2f}", ha="center", va="center", fontsize=6.5, fontweight="bold", color=tc)
    full_box(ax_g, 0.85)
    ax_g.tick_params(length=0)
    ax_g.add_patch(mpatches.Rectangle((4, 4), 2, 2, fill=False, edgecolor="#B91C1C", ls="--", lw=1.45, zorder=6))
    ax_g.set_xticklabels(LINEAGE_SHORT[:len(ord_l)], rotation=38, ha="right", fontsize=7.4, fontweight="bold")
    ax_g.set_yticklabels(LINEAGE_SHORT[:len(ord_l)], rotation=0, fontsize=7.4, fontweight="bold")
    ax_g.set_xlabel("TOLT RNA+ATAC predicted", fontsize=8.6); ax_g.set_ylabel("Measured gate", fontsize=8.6)
    ax_g.set_title("9-lineage confusion matrix (79.3% acc.)", fontsize=9.3)
    plabel(ax_g, "g", x=-0.28, y=1.05)

    # Panel h: Few-Shot Cross-Well & Cross-Donor 9-Lineage Flow-Gating Stability
    ax_h = fig.add_subplot(gs[2, 9:12])
    style_precision_ax(ax_h, cat_x=True, grid="y")
    m_h = ["scVAEIT_RNA+ATAC", "Classical_RNA+ATAC", "TOLT_RNA", "TOLT_RNA+ATAC"]
    lbl_h = ["scVAEIT", "SVD\nDual", "TOLT\nRNA", "TOLT\nDual"]
    xh = np.arange(len(m_h))
    v_cw = [float(df_sum[(df_sum["transfer_tier"] == "Cross_Well_Same_Donor_HMN200910") & (df_sum["budget"].astype(str) == "250") & (df_sum["method"] == m)]["pheno9_p1_acc_pct"].mean()) for m in m_h]
    v_cd = [float(df_sum[(df_sum["transfer_tier"] == "True_Cross_Donor_Triomic") & (df_sum["budget"].astype(str) == "250") & (df_sum["method"] == m)]["pheno9_p1_acc_pct"].mean()) for m in m_h]
    ax_h.axhline(v_cd[0], color="#6D28D9", ls=":", lw=0.80, zorder=2)
    ax_h.bar(xh - 0.18, v_cw, width=0.35, color="#1D4ED8", edgecolor=INK, lw=0.80, label="Cross-Well ($n=250$)", zorder=3)
    ax_h.bar(xh + 0.18, v_cd, width=0.35, color="#B91C1C", edgecolor=INK, lw=0.80, label="Cross-Donor ($n=250$)", zorder=3)
    ax_h.set_xticks(xh); ax_h.set_xticklabels(lbl_h, fontsize=8.2, fontweight="bold")
    ax_h.set_ylim(0, 84)
    ax_h.set_ylabel("9-lineage flow accuracy (%)")
    ax_h.set_title("Few-shot gating stability")
    ax_h.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.0, loc="upper left")
    plabel(ax_h, "h", x=-0.18)

    save_fig(fig, MAIN_FIG_DIR, "Figure7")


# =============================================================================
# FIGURE 8: Tri-Omic TOLT vs 30M-33M Unimodal Foundation Models (scGPT & Geneformer) (a-g)
# =============================================================================
def make_figure8_foundation_models() -> None:
    df_fm = pd.read_csv(TABLES_DIR / "v42_foundation_model_comparison.csv")
    df_dm = pd.read_csv(TABLES_DIR / "v42_foundation_decoupled_markers.csv")
    df_tp = pd.read_csv(TABLES_DIR / "v42_foundation_teaseq_phenotyping.csv")
    df_st = pd.read_csv(TABLES_DIR / "v42_foundation_steady_throughput.csv")

    df_fm["primary_pcc"] = df_fm["expressed_pcc_median"].fillna(df_fm["protein_pcc_median"])

    fig = plt.figure(figsize=(12.6, 8.5))
    gs = gridspec.GridSpec(3, 12, figure=fig, height_ratios=[0.96, 1.06, 1.00], hspace=0.34, wspace=0.88)

    # Panel a: Architectural & Pretraining Paradigm Matrix (TOLT vs scGPT vs Geneformer)
    ax_a = fig.add_subplot(gs[0, 0:4])
    spec_txt = np.array([
        ["0.33M\n(101× less)", "31.0M", "RNA+ATAC\n+ADT", "137-tok\nO(L)", "Native\nGNLL σ"],
        ["33.0M\nhuman", "50.8M", "RNA\nonly", "1200-tok\nO(L²)", "None"],
        ["30.0M\nhuman", "10.3M", "RNA\nonly", "2048-tok\nO(L²)", "None"],
    ])
    spec_score = np.array([
        [0.95, 0.78, 0.98, 0.95, 0.95],
        [0.25, 0.45, 0.35, 0.30, 0.15],
        [0.28, 0.88, 0.35, 0.25, 0.15],
    ])
    sns.heatmap(
        spec_score, annot=False, cmap=PUBCNS_DIV, vmin=0.0, vmax=1.0, cbar=False,
        xticklabels=["Pretrain\nCorpus", "Model\nParams", "Modalities\nAligned", "Attention\nScaling", "Per-Cell\nUncert."],
        yticklabels=["TOLT\n(Tri-Omic)", "scGPT\n(33M RNA)", "Geneformer\n(30M RNA)"],
        linewidths=0.85, linecolor=INK, ax=ax_a,
    )
    for ri in range(spec_score.shape[0]):
        for ci in range(spec_score.shape[1]):
            sv = spec_score[ri, ci]
            tc = "white" if (sv > 0.72 or sv < 0.24) else INK
            ax_a.text(ci + 0.5, ri + 0.5, spec_txt[ri, ci], ha="center", va="center", fontsize=7.3, fontweight="bold", color=tc)
    full_box(ax_a, 0.85)
    ax_a.tick_params(length=0)
    ax_a.set_xticklabels(ax_a.get_xticklabels(), fontsize=7.7, fontweight="bold")
    ax_a.set_yticklabels(ax_a.get_yticklabels(), rotation=0, fontsize=8.0, fontweight="bold")
    ax_a.set_title("Foundation pretraining & architecture design", fontsize=9.3)
    plabel(ax_a, "a", x=-0.20)

    # Panel b: Universal Lift of Cross-Fitted OOF Delta^2-Simplex Calibration Across All 4 Cohorts (Full/6000)
    ax_b = fig.add_subplot(gs[0, 4:8])
    style_precision_ax(ax_b, cat_x=True, grid="y")
    coh_cfg = [
        ("GSE194122_BMMC_CrossSite", "full", "GSE194122\nCross-Site"),
        ("GSE128639_BMMC_Stuart", "full", "GSE128639\nDonor A↔B"),
        ("GSE166489_188AB_LODO", "6000", "GSE166489\n5-Fold LODO"),
        ("GSE158013_TEAseq_CrossDonor", "full", "GSE158013\nTEA-seq"),
    ]
    xb = np.arange(len(coh_cfg))
    wb = 0.122
    for ci, (ckey, b_full, _) in enumerate(coh_cfg):
        if ci % 2 == 1:
            ax_b.axvspan(ci - 0.46, ci + 0.46, color="#F8FAFC", zorder=0)
        if ci > 0:
            ax_b.axvline(ci - 0.5, color="#94A3B8", ls="--", lw=0.60, zorder=1)
        sub_c = df_fm[(df_fm["cohort"] == ckey) & (df_fm["budget"].astype(str) == b_full)]
        v_svd = float(sub_c[sub_c["regime"] == "SVD-Ridge"]["primary_pcc"].mean())
        ax_b.plot([ci - 0.40, ci + 0.40], [v_svd, v_svd], color="#334155", ls="--", lw=1.30, zorder=5, label="Local SVD-Ridge" if ci == 0 else "")
        for bi, (bnm, bcol) in enumerate([("Geneformer", "#047857"), ("scGPT", "#5B21B6"), ("TOLT", "#B91C1C")]):
            b_match = "TOLT_RNA+ATAC" if (ckey == "GSE158013_TEAseq_CrossDonor" and bnm == "TOLT") else bnm
            vd = float(sub_c[(sub_c["backbone"] == b_match) & (sub_c["regime"] == "Direct_Linear_Probe")]["primary_pcc"].mean())
            va = float(sub_c[(sub_c["backbone"] == b_match) & (sub_c["regime"] == "OOF_Simplex_Adapter")]["primary_pcc"].mean())
            xp_d = ci + (bi * 2 - 2.5) * wb
            xp_a = ci + (bi * 2 - 1.5) * wb
            ax_b.bar(xp_d, vd, width=wb * 0.92, color=bcol, alpha=0.45, edgecolor=INK, lw=0.70, hatch="//", label=f"{bnm} Direct" if ci == 0 else "", zorder=3)
            ax_b.bar(xp_a, va, width=wb * 0.92, color=bcol, edgecolor=INK, lw=0.80, label=f"{bnm} + OOF Δ²" if ci == 0 else "", zorder=4)
    ax_b.set_xticks(xb); ax_b.set_xticklabels([c[2] for c in coh_cfg], fontsize=7.9, fontweight="bold")
    ax_b.set_ylim(0.38, 1.10)
    ax_b.set_ylabel("Primary median PCC (Full / 6k)")
    ax_b.set_title("Universal OOF $\\Delta^2$-simplex calibration lift", fontsize=9.3)
    ax_b.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=6.8, loc="upper left")
    plabel(ax_b, "b", x=-0.14)

    # Panel c: Extreme Few-Shot Sample Complexity (n_ref=50 -> Full) on Cross-Site GSE194122
    ax_c = fig.add_subplot(gs[0, 8:12])
    sub_90 = df_fm[df_fm["cohort"] == "GSE194122_BMMC_CrossSite"]
    ax_c.axvline(250.0, color="#B91C1C", ls="--", lw=0.85, zorder=2)
    svd_90_full = float(sub_90[(sub_90["method"] == "SVD-Ridge") & (sub_90["budget"].astype(str) == "full")]["primary_pcc"].mean())
    ax_c.axhline(svd_90_full, color="#475569", ls=":", lw=0.80, zorder=2)
    ax_c.text(54.0, 0.772, "Ultra-low $n_{\\mathrm{ref}}$: TOLT #1 ($n=50$)", fontsize=7.1, fontweight="bold", color="#B91C1C", va="top",
              bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.55}, zorder=6)
    curves_c = [
        ("TOLT__OOF_Simplex_Adapter", "#B91C1C", "o", "-", "TOLT + OOF Δ² (0.33M)"),
        ("scGPT__OOF_Simplex_Adapter", "#5B21B6", "s", "--", "scGPT + OOF Δ² (33M)"),
        ("SVD-Ridge", "#475569", "D", "-.", "Local SVD-Ridge"),
        ("Geneformer__OOF_Simplex_Adapter", "#047857", "^", "-", "Geneformer + OOF Δ² (30M)"),
        ("Geneformer__Direct_Linear_Probe", "#94A3B8", "v", ":", "Geneformer Direct"),
    ]
    for mtag, col, mk, ls, lbl in curves_c:
        gm = sub_90[sub_90["method"] == mtag].groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), pcc=("primary_pcc", "mean")).sort_values("n_ref")
        if len(gm):
            ax_c.plot(gm["n_ref"], gm["pcc"], marker=mk, ls=ls, lw=2.2 if mtag.startswith("TOLT") else 1.45,
                      markersize=4.7, color=col, markeredgecolor=INK, markeredgewidth=0.65, label=lbl, zorder=5 if mtag.startswith("TOLT") else 3)
    ax_c.set_xscale("log")
    style_precision_ax(ax_c, log_x=True, grid="both")
    ax_c.set_ylim(0.43, 0.782)
    ax_c.set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$, log scale)")
    ax_c.set_ylabel("Expressed-receptor median PCC")
    ax_c.set_title("Cross-site sample complexity (GSE194122)", fontsize=9.3)
    ax_c.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.5, loc="lower right")
    plabel(ax_c, "c", x=-0.14)

    # Panel d: Spliced, Post-Transcriptionally Decoupled & High-Dropout Surface Receptor Showdown
    ax_d = fig.add_subplot(gs[1, 0:6])
    style_precision_ax(ax_d, cat_x=True, grid="y")
    mks_show = [
        ("CD45RA", "CD45RA\n(splice)"),
        ("CD95", "CD95\n(FAS)"),
        ("CD4", "CD4\n(low-UMI)"),
        ("CD279", "PD-1\n(0-UMI)"),
        ("CD56", "CD56\n(NCAM1)"),
        ("CD82", "CD82\n(tetrasp.)"),
    ]
    dm_500 = df_dm[df_dm["budget"].astype(str) == "500"]
    xd = np.arange(len(mks_show))
    wd = 0.192
    m_dm_cfg = [
        ("Geneformer__OOF_Simplex_Adapter", "#047857", "Geneformer (30M RNA)"),
        ("SVD-Ridge", "#475569", "Local SVD-Ridge"),
        ("scGPT__OOF_Simplex_Adapter", "#5B21B6", "scGPT (33M RNA)"),
        ("TOLT__OOF_Simplex_Adapter", "#B91C1C", "TOLT (0.33M Tri-Omic)"),
    ]
    ax_d.axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    for xi in range(len(mks_show)):
        if xi % 2 == 1:
            ax_d.axvspan(xi - 0.46, xi + 0.46, color="#F8FAFC", zorder=0)
        if xi > 0:
            ax_d.axvline(xi - 0.5, color="#94A3B8", ls="--", lw=0.60, zorder=1)
    for mi, (m_code, m_col, m_lbl) in enumerate(m_dm_cfg):
        means, sems = [], []
        for xi, (ab_k, _) in enumerate(mks_show):
            v = dm_500[(dm_500["antibody"] == ab_k) & (dm_500["method"] == m_code)]["pcc"].to_numpy(dtype=float)
            means.append(float(v.mean()) if len(v) else 0.0)
            sems.append(float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
            if m_code in {"TOLT__OOF_Simplex_Adapter", "scGPT__OOF_Simplex_Adapter"} and len(v):
                jit = np.linspace(-0.022, 0.022, len(v))
                ax_d.scatter(xi + (mi - 1.5) * wd + jit, v, s=11, facecolor="white", edgecolor=INK, lw=0.55, zorder=5)
        ax_d.bar(xd + (mi - 1.5) * wd, means, width=wd, yerr=sems, capsize=1.6, error_kw={"lw": 0.65, "ecolor": INK},
                 color=m_col, edgecolor=INK, lw=0.78, label=m_lbl, zorder=3)
    sig_bracket(ax_d, 0 + 0.5 * wd, 0 + 1.5 * wd, 0.82, 0.02, "+0.03 vs scGPT", fontsize=7.1)
    sig_bracket(ax_d, 3 + 0.5 * wd, 3 + 1.5 * wd, 0.62, 0.02, "+0.04 vs scGPT", fontsize=7.1)
    ax_d.set_xticks(xd); ax_d.set_xticklabels([m[1] for m in mks_show], fontsize=7.9, fontweight="bold")
    ax_d.set_ylim(0.25, 1.01)
    ax_d.set_ylabel("5-donor LODO Pearson $r$ ($n_{\\mathrm{ref}}=500$)")
    ax_d.set_title("Spliced, low-transcript & high-dropout surface receptors", fontsize=9.2)
    ax_d.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=7.2, loc="upper right")
    plabel(ax_d, "d", x=-0.09)

    # Panel e: 188-Antibody LODO Global MSE Calibration (n_ref=250, 500, 6000)
    ax_e = fig.add_subplot(gs[1, 7:12])
    style_precision_ax(ax_e, cat_x=True, grid="y")
    sub_lo = df_fm[df_fm["cohort"] == "GSE166489_188AB_LODO"]
    b_ord_e = ["250", "500", "6000"]
    xe = np.arange(len(b_ord_e))
    for m_code, m_col, mk, ls, lbl in [
        ("TOLT__OOF_Simplex_Adapter", "#B91C1C", "o", "-", "TOLT + OOF Δ²"),
        ("scGPT__OOF_Simplex_Adapter", "#5B21B6", "s", "--", "scGPT + OOF Δ²"),
        ("Geneformer__OOF_Simplex_Adapter", "#047857", "^", "-", "Geneformer + OOF Δ²"),
        ("SVD-Ridge", "#475569", "D", "-.", "SVD-Ridge"),
        ("Geneformer__Direct_Linear_Probe", "#94A3B8", "v", ":", "Geneformer Direct"),
    ]:
        mv = [float(sub_lo[(sub_lo["budget"].astype(str) == b) & (sub_lo["method"] == m_code)]["mse"].mean()) for b in b_ord_e]
        ax_e.plot(xe, mv, marker=mk, ls=ls, lw=2.2 if m_code.startswith("TOLT") else 1.45, markersize=4.9,
                  color=m_col, markeredgecolor=INK, markeredgewidth=0.65, label=lbl, zorder=5 if m_code.startswith("TOLT") else 3)
    ax_e.set_xticks(xe); ax_e.set_xticklabels(["$n_{\\mathrm{ref}}=250$", "$n_{\\mathrm{ref}}=500$", "$n_{\\mathrm{ref}}=6{,}000$"], fontweight="bold")
    ax_e.set_ylim(0.095, 0.172)
    ax_e.set_ylabel("188-target panel MSE (lower is better)")
    ax_e.set_title("188-plex LODO error calibration", fontsize=9.3)
    ax_e.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.0, loc="upper right")
    plabel(ax_e, "e", x=-0.14)

    # Panel f: Cross-Donor Trimodal TEA-seq (X066_W3 <-> X061_W1) Chromatin Fusion vs Unimodal RNA Foundation Models
    ax_f = fig.add_subplot(gs[2, 0:7])
    style_precision_ax(ax_f, cat_x=True, grid="y")
    tea_f = df_fm[df_fm["cohort"] == "GSE158013_TEAseq_CrossDonor"]
    tp_full = df_tp[df_tp["budget"].astype(str) == "full"]
    tp_250 = df_tp[df_tp["budget"].astype(str) == "250"]
    m_tea_cfg = [
        ("Geneformer__OOF_Simplex_Adapter", "#047857", "Geneformer (RNA)"),
        ("SVD-Ridge", "#475569", "SVD-Ridge (RNA+ATAC)"),
        ("TOLT_RNA__OOF_Simplex_Adapter", "#1D4ED8", "TOLT (RNA only)"),
        ("scGPT__OOF_Simplex_Adapter", "#5B21B6", "scGPT (33M RNA)"),
        ("TOLT_RNA+ATAC__OOF_Simplex_Adapter", "#B91C1C", "TOLT (RNA+ATAC)"),
    ]
    xf = np.arange(4)
    wf = 0.158
    ax_f.axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    for xi in range(4):
        if xi % 2 == 1:
            ax_f.axvspan(xi - 0.46, xi + 0.46, color="#F8FAFC", zorder=0)
        if xi > 0:
            ax_f.axvline(xi - 0.5, color="#94A3B8", ls="--", lw=0.60, zorder=1)
    for mi, (m_k, m_c, m_l) in enumerate(m_tea_cfg):
        v1 = float(tea_f[(tea_f["budget"].astype(str) == "500") & (tea_f["method"] == m_k)]["primary_pcc"].mean())
        v2 = float(tea_f[(tea_f["budget"].astype(str) == "full") & (tea_f["method"] == m_k)]["primary_pcc"].mean())
        v3 = float(tp_250[tp_250["method"] == m_k]["p2_macro_f1_8lin"].mean())
        v4 = float(tp_full[tp_full["method"] == m_k]["p1_accuracy_pct"].mean()) / 100.0
        vals_f = [v1, v2, v3, v4]
        ax_f.bar(xf + (mi - 2) * wf, vals_f, width=wf, color=m_c, edgecolor=INK, lw=0.78, label=m_l, zorder=3)
    ax_f.set_xticks(xf)
    ax_f.set_xticklabels(["45-ADT PCC\n($n_{\\mathrm{ref}}=500$)", "45-ADT PCC\n(Full Ref.)", "9-Lin. Macro-F1\n($n_{\\mathrm{ref}}=250$)", "9-Lin. Gate Acc.\n(Full Ref.)"], fontsize=7.9, fontweight="bold")
    ax_f.set_ylim(0.32, 0.65)
    ax_f.set_ylabel("Cross-donor trimodal score")
    ax_f.set_title("Open-chromatin fusion outperforms unimodal RNA foundation models on TEA-seq", fontsize=9.3)
    ax_f.legend(frameon=True, facecolor="white", edgecolor=INK, ncol=3, fontsize=6.9, loc="upper left")
    plabel(ax_f, "f", x=-0.09)

    # Panel g: Steady-State GPU Inference Throughput on Consumer 8 GB RTX 5070 Laptop
    ax_g = fig.add_subplot(gs[2, 8:12])
    style_precision_ax(ax_g, cat_y=True, grid="x")
    st_ord = df_st.set_index("model").loc[["scGPT_human", "Geneformer_30M", "TOLT-Base-110"]].reset_index()
    yg = np.arange(len(st_ord))
    cols_g = ["#5B21B6", "#047857", "#B91C1C"]
    lbls_g = ["scGPT\n(50.8M, 1200-tok)", "Geneformer\n(10.3M, 2048-tok)", "TOLT\n(31.0M, 137-tok)"]
    ax_g.axvline(float(st_ord.loc[0, "cells_per_sec"]), color="#5B21B6", ls=":", lw=0.80, zorder=2)
    ax_g.barh(yg, st_ord["cells_per_sec"], height=0.54, color=cols_g, edgecolor=INK, lw=0.85, zorder=3)
    for ri, r in st_ord.iterrows():
        spd = float(r["cells_per_sec"])
        ax_g.text(spd + 22, int(ri), f"{spd:.0f} cells/s\n({r['gpu_inference_sec']:.1f}s)", va="center", fontsize=7.5, fontweight="bold")
    ax_g.set_yticks(yg); ax_g.set_yticklabels(lbls_g, fontsize=7.9, fontweight="bold")
    ax_g.set_xlim(0, 1180)
    ax_g.set_xlabel("Steady-state BF16 throughput (cells/s)")
    ax_g.set_title("52× faster inference vs scGPT", fontsize=9.3)
    plabel(ax_g, "g", x=-0.28)

    save_fig(fig, MAIN_FIG_DIR, "Figure8")


# =============================================================================
# SUPPLEMENTARY FIGURES S1-S6 (Complete Extended Data Suite)
# =============================================================================
def make_supplementary_figures() -> None:
    s_diag = np.load(FIGURE_PRED_DIR / "v42_supp_diagnostics.npz", allow_pickle=True)

    # --- Figure S1 (a-d): Euclidean Distance Concentration Collapse & Pairwise Ratio Distribution ---
    df_lc = pd.read_csv(TABLES_DIR / "benchmark_cross_donor_learning_curves.csv")
    df_lo = pd.read_csv(TABLES_DIR / "v41_lodo_paired_summary.csv")
    fig1, axes1_grid = plt.subplots(2, 2, figsize=(10.8, 6.6))
    axes1 = axes1_grid.ravel()
    plt.subplots_adjust(hspace=0.33, wspace=0.25)
    for ax, coh, ttl, ltr in [
        (axes1[0], "GSE194122_BMMC_CrossSite", "GSE194122 Cross-Site (134 ADTs)", "a"),
        (axes1[1], "GSE128639_BMMC_Stuart", "GSE128639 Cross-Donor (25 ADTs)", "b"),
    ]:
        ax.axvline(500.0, color="#B91C1C", ls="--", lw=0.75, zorder=2)
        sub = df_lc[df_lc["cohort"] == coh]
        for m, ls, mk in [("TOLT", "-", "o"), ("Cosine-kNN", "--", "s"), ("SVD-Ridge", "-.", "^"), ("Legacy-Euclid-kNN256", ":", "v")]:
            gm = sub[sub["method"] == m].groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), pcc=("protein_pcc_median", "mean")).sort_values("n_ref")
            if len(gm):
                ax.plot(gm["n_ref"], gm["pcc"], ls=ls, marker=mk, lw=1.85, color=PALETTE[m], markeredgecolor=INK, markeredgewidth=0.65, label=m, zorder=4 if m == "TOLT" else 3)
        ax.set_xscale("log")
        style_precision_ax(ax, log_x=True, grid="both")
        ax.set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$)"); ax.set_ylabel("Median antibody PCC")
        ax.set_title(ttl, fontsize=9.3); ax.legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.9)
        plabel(ax, ltr)

    axes1[2].axvline(500.0, color="#B91C1C", ls="--", lw=0.75, zorder=2)
    for m, ls, mk in [("TOLT", "-", "o"), ("Cosine-kNN", "--", "s"), ("SVD-Ridge", "-.", "^"), ("Legacy-Euclid-kNN256", ":", "v")]:
        gm = df_lo[df_lo["method"] == m].groupby("budget", as_index=False).agg(n_ref=("n_ref", "mean"), pcc=("expressed_protein_pcc_median", "mean")).sort_values("n_ref")
        if len(gm):
            axes1[2].plot(gm["n_ref"], gm["pcc"], ls=ls, marker=mk, lw=1.85, color=PALETTE[m], markeredgecolor=INK, markeredgewidth=0.65, label=m, zorder=4 if m == "TOLT" else 3)
    axes1[2].set_xscale("log")
    style_precision_ax(axes1[2], log_x=True, grid="both")
    axes1[2].set_xlabel("Paired reference cells ($n_{\\mathrm{ref}}$)"); axes1[2].set_ylabel("Expressed median PCC")
    axes1[2].set_title("GSE166489 188-AB 5-Fold LODO", fontsize=9.3); axes1[2].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.9)
    plabel(axes1[2], "c")

    style_precision_ax(axes1[3], grid="both")
    r_euc = s_diag["s1_ratio_euc256"]
    r_cos = s_diag["s1_ratio_cos50"]
    axes1[3].hist(r_cos, bins=30, density=True, color="#1D4ED8", alpha=0.85, edgecolor=INK, lw=0.50, label=f"PCA50 Cosine (med {np.median(r_cos):.2f})", zorder=3)
    axes1[3].hist(r_euc, bins=30, density=True, color="#B91C1C", alpha=0.85, edgecolor=INK, lw=0.50, label=f"256-d Euclid. (med {np.median(r_euc):.2f})", zorder=4)
    axes1[3].axvline(float(np.median(r_cos)), color="#1D4ED8", ls="--", lw=0.90, zorder=5)
    axes1[3].axvline(float(np.median(r_euc)), color="#B91C1C", ls="--", lw=0.90, zorder=5)
    axes1[3].axvline(1.0, color=INK, ls=":", lw=0.95, zorder=5)
    axes1[3].set_xlabel("Nearest-to-median distance ratio ($d_{(1)} / d_{\\mathrm{med}}$)")
    axes1[3].set_ylabel("Query cell density")
    axes1[3].set_title("Empirical distance concentration", fontsize=9.3)
    axes1[3].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=6.9, loc="upper left")
    plabel(axes1[3], "d")
    save_fig(fig1, SUPP_FIG_DIR, "FigureS1")

    # --- Figure S2 (a-d): GSE166489 Panel Hygiene, Clone-Swap & Single-Cell Fc-Binding Scatter ---
    df_hyg = pd.read_csv(TABLES_DIR / "v41_panel_hygiene_audit.csv")
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 7.1))
    plt.subplots_adjust(hspace=0.30, wspace=0.25)
    xh = np.arange(len(df_hyg))
    style_precision_ax(axes2[0, 0], cat_x=True, grid="y")
    axes2[0, 0].axhline(0.95, color="#64748B", ls=":", lw=0.75, zorder=2)
    axes2[0, 0].bar(xh - 0.18, df_hyg["measured_CD8_vs_CD8a_pcc"], width=0.35, color="#1D4ED8", edgecolor=INK, lw=0.80, label="Measured CD8 vs CD8a", zorder=3)
    axes2[0, 0].bar(xh + 0.18, df_hyg["predicted_CD8_vs_CD8a_pcc"], width=0.35, color="#B91C1C", edgecolor=INK, lw=0.80, label="TOLT LODO CD8 vs CD8a", zorder=3)
    axes2[0, 0].set_xticks(xh); axes2[0, 0].set_xticklabels(df_hyg["sample_id"], fontweight="bold")
    axes2[0, 0].set_ylim(0.85, 1.015); axes2[0, 0].set_ylabel("Dual-clone Pearson $r$")
    axes2[0, 0].set_title("Dual CD8 (0046) & CD8a (0080) clone concordance")
    axes2[0, 0].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.5)
    plabel(axes2[0, 0], "a")

    style_precision_ax(axes2[0, 1], cat_x=True, grid="y")
    axes2[0, 1].axhline(0.50, color="#64748B", ls=":", lw=0.75, zorder=2)
    axes2[0, 1].bar(xh - 0.22, df_hyg["measured_TCRgd_vs_TCRVg9_0582_pcc"], width=0.22, color="#047857", edgecolor=INK, lw=0.75, label="TCRgd vs corrected TCRVg9_0582", zorder=3)
    axes2[0, 1].bar(xh, df_hyg["measured_TCRgd_vs_TCRVg_0583_pcc"], width=0.22, color="#5B21B6", edgecolor=INK, lw=0.75, label="TCRgd vs corrected TCRVg_0583", zorder=3)
    axes2[0, 1].bar(xh + 0.22, df_hyg["excised_IgGFc_vs_CD64_fc_binding_pcc"], width=0.22, color="#B45309", edgecolor=INK, lw=0.75, label="Excised IgGFc vs FcγRI (CD64)", zorder=3)
    axes2[0, 1].set_xticks(xh); axes2[0, 1].set_xticklabels(df_hyg["sample_id"], fontweight="bold")
    axes2[0, 1].set_ylim(0, 0.90); axes2[0, 1].set_ylabel("Pearson correlation ($r$)")
    axes2[0, 1].set_title("Upstream TCRVγ9/Vγ swap verification & IgGFc Fc-binding")
    axes2[0, 1].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.2)
    plabel(axes2[0, 1], "b")

    style_precision_ax(axes2[1, 0], grid="both")
    c8_1, c8_2 = s_diag["s2_p11_cd8_0046"], s_diag["s2_p11_cd8a_0080"]
    r_c8 = float(np.corrcoef(c8_1, c8_2)[0, 1])
    axes2[1, 0].scatter(c8_1[::2], c8_2[::2], s=4.0, c="#1D4ED8", alpha=0.52, linewidths=0, rasterized=True, zorder=2)
    axes2[1, 0].plot([0, 4.2], [0, 4.2], color="#B91C1C", ls="--", lw=1.20, zorder=3)
    axes2[1, 0].text(0.05, 0.92, f"MIS-C P1.1 ($n=9{{,}}264$)\nPearson $r = {r_c8:.3f}$", transform=axes2[1, 0].transAxes, va="top",
                     fontsize=8.0, fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": INK, "lw": 0.65})
    axes2[1, 0].set_xlabel("Measured CD8 clone 0046 (CLR)"); axes2[1, 0].set_ylabel("Measured CD8a clone 0080 (CLR)")
    axes2[1, 0].set_title("Single-cell anti-CD8 dual-clone linearity")
    plabel(axes2[1, 0], "c")

    style_precision_ax(axes2[1, 1], grid="both")
    c64, igg, c14 = s_diag["s2_p11_cd64"], s_diag["s2_p11_iggfc"], s_diag["s2_p11_cd14"]
    r_fc = float(np.corrcoef(c64, igg)[0, 1])
    sl_fc, ic_fc = np.polyfit(c64, igg, 1)
    xfc = np.linspace(float(c64.min()), float(c64.max()), 50)
    sc_fc = axes2[1, 1].scatter(c64[::2], igg[::2], c=c14[::2], cmap=PUBCNS_DIV, vmin=0, vmax=max(float(np.percentile(c14, 98.5)), 1.0), s=5.5, alpha=0.80, linewidths=0, rasterized=True, zorder=2)
    axes2[1, 1].plot(xfc, sl_fc * xfc + ic_fc, color=INK, ls="--", lw=1.15, zorder=4)
    cb_fc = fig2.colorbar(sc_fc, ax=axes2[1, 1], pad=0.02, fraction=0.045)
    cb_fc.outline.set_edgecolor(INK); cb_fc.outline.set_linewidth(0.75)
    cb_fc.set_label("CD14 CLR", fontsize=7.9)
    axes2[1, 1].text(0.05, 0.92, f"Non-specific FcγRI binding\nPearson $r = {r_fc:.3f}$", transform=axes2[1, 1].transAxes, va="top",
                     fontsize=8.0, fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#B91C1C", "lw": 0.65})
    axes2[1, 1].set_xlabel("Monocyte FcγRI / CD64 (CLR)"); axes2[1, 1].set_ylabel("Excised IgGFc control (CLR)")
    axes2[1, 1].set_title("Inflammatory monocyte FcγRI drives IgGFc artifact")
    plabel(axes2[1, 1], "d")
    save_fig(fig2, SUPP_FIG_DIR, "FigureS2")

    # --- Figure S3 (a-d): Abundance-Tier Simplex Weights & Multi-Cohort Sample-Complexity Matrices ---
    df_cp = pd.read_csv(TABLES_DIR / "gse166489_per_antibody_coupling.csv")
    df_mc = pd.read_csv(TABLES_DIR / "v41_multiseed_learning_curves.csv")
    fig3, axes3 = plt.subplots(2, 2, figsize=(11.55, 6.55))
    plt.subplots_adjust(hspace=0.30, wspace=0.28)

    style_precision_ax(axes3[0, 0], cat_x=True, grid="y")
    w_t = df_cp.groupby("tier", as_index=False)[["w_local_svd", "w_tolt_direct", "w_tolt_residual"]].mean().set_index("tier").loc[["Tier1", "Tier2", "Tier3"]].reset_index()
    xt = np.arange(3)
    axes3[0, 0].axhline(0.50, color=INK, ls="--", lw=0.75, zorder=2)
    axes3[0, 0].bar(xt, w_t["w_local_svd"], width=0.50, color="#64748B", edgecolor=INK, lw=0.80, label="Local SVD ($w_{\\mathrm{loc}}^{\\star}$)", zorder=3)
    axes3[0, 0].bar(xt, w_t["w_tolt_direct"], bottom=w_t["w_local_svd"], width=0.50, color="#1D4ED8", edgecolor=INK, lw=0.80, label="Direct TOLT ($w_{\\mathrm{dir}}^{\\star}$)", zorder=3)
    axes3[0, 0].bar(xt, w_t["w_tolt_residual"], bottom=w_t["w_local_svd"] + w_t["w_tolt_direct"], width=0.50, color="#B91C1C", edgecolor=INK, lw=0.80, label="Residual ($w_{\\mathrm{cor}}^{\\star}$)", zorder=3)
    axes3[0, 0].set_xticks(xt); axes3[0, 0].set_xticklabels(["Tier 1\nExpressed", "Tier 2\nRare", "Tier 3\nBackground"], fontweight="bold")
    axes3[0, 0].set_ylim(0, 1.30); axes3[0, 0].set_ylabel("Mean OOF $\\Delta^2$ simplex weight")
    axes3[0, 0].set_title("Simplex allocation across abundance tiers")
    axes3[0, 0].legend(frameon=True, facecolor="white", edgecolor=INK, ncol=1, fontsize=7.3, loc="upper right")
    plabel(axes3[0, 0], "a")

    m_hm = ["TOLT", "TOLT-Residual", "scLinear", "ExtraTrees", "Cosine-kNN", "SVD-Ridge", "Seurat-Anchor", "totalVI", "sciPENN"]
    b_hm = ["50", "100", "250", "500", "1000", "full"]
    for ax_hm_s, coh_s, vcol_s, vmin_s, vmax_s, ttl_s, ltr_s in [
        (axes3[0, 1], "GSE194122_BMMC_CrossSite", "expressed_pcc_median", 0.48, 0.75, "GSE194122 cross-site matrix (134 ADTs)", "b"),
        (axes3[1, 0], "GSE128639_BMMC_Stuart", "protein_pcc_median", 0.52, 0.88, "GSE128639 cross-donor matrix (25 ADTs)", "c"),
    ]:
        piv_s = df_mc[(df_mc["cohort"] == coh_s) & (df_mc["method"].isin(m_hm)) & (df_mc["budget"].isin(b_hm))].groupby(["method", "budget"], as_index=False)[vcol_s].mean().pivot(index="method", columns="budget", values=vcol_s).reindex(index=m_hm, columns=b_hm)
        mat_s = piv_s.to_numpy(dtype=float)
        sns.heatmap(piv_s, annot=False, cmap=PUBCNS_DIV, vmin=vmin_s, vmax=vmax_s, linewidths=0.60, linecolor=INK,
                    cbar_kws={"label": "Median PCC", "pad": 0.02}, ax=ax_hm_s)
        for ri in range(mat_s.shape[0]):
            for ci in range(mat_s.shape[1]):
                if not np.isnan(mat_s[ri, ci]):
                    nv = (mat_s[ri, ci] - vmin_s) / (vmax_s - vmin_s)
                    tc = "white" if (nv > 0.72 or nv < 0.24) else INK
                    ax_hm_s.text(ci + 0.5, ri + 0.5, f"{mat_s[ri, ci]:.3f}", ha="center", va="center", fontsize=7.1, fontweight="bold", color=tc)
        full_box(ax_hm_s, 0.85)
        ax_hm_s.tick_params(length=0)
        ax_hm_s.set_yticklabels(m_hm, rotation=0, fontweight="bold", fontsize=7.7)
        ax_hm_s.set_xlabel("Reference budget $n_{\\mathrm{ref}}$"); ax_hm_s.set_ylabel("")
        ax_hm_s.set_title(ttl_s)
        plabel(ax_hm_s, ltr_s)

    b_lo = ["100", "250", "500", "6000"]
    piv_lo = df_lo[(df_lo["method"].isin(m_hm)) & (df_lo["budget"].astype(str).isin(b_lo))].assign(b_str=lambda d: d["budget"].astype(str)).groupby(["method", "b_str"], as_index=False)["expressed_protein_pcc_median"].mean().pivot(index="method", columns="b_str", values="expressed_protein_pcc_median").reindex(index=m_hm, columns=b_lo)
    mat_lo = piv_lo.to_numpy(dtype=float)
    sns.heatmap(piv_lo, annot=False, cmap=PUBCNS_DIV, vmin=0.40, vmax=0.66, linewidths=0.60, linecolor=INK,
                cbar_kws={"label": "Expressed PCC", "pad": 0.02}, ax=axes3[1, 1])
    for ri in range(mat_lo.shape[0]):
        for ci in range(mat_lo.shape[1]):
            if not np.isnan(mat_lo[ri, ci]):
                nv = (mat_lo[ri, ci] - 0.40) / (0.66 - 0.40)
                tc = "white" if (nv > 0.72 or nv < 0.24) else INK
                axes3[1, 1].text(ci + 0.5, ri + 0.5, f"{mat_lo[ri, ci]:.3f}", ha="center", va="center", fontsize=7.3, fontweight="bold", color=tc)
    full_box(axes3[1, 1], 0.85)
    axes3[1, 1].tick_params(length=0)
    axes3[1, 1].set_yticklabels(m_hm, rotation=0, fontweight="bold", fontsize=7.7)
    axes3[1, 1].set_xlabel("Reference budget $n_{\\mathrm{ref}}$ (5-donor LODO)"); axes3[1, 1].set_ylabel("")
    axes3[1, 1].set_title("GSE166489 5-fold LODO matrix (188 ADTs)")
    plabel(axes3[1, 1], "d")
    save_fig(fig3, SUPP_FIG_DIR, "FigureS3")

    # --- Figure S4 (a-n): Complete 14-Marker 1D CLR Bimodal 2-GMM Density Gallery (3x5 Grid) ---
    mk_names_14 = [str(x) for x in s_diag["s4_marker_names"]]
    xg_14 = s_diag["s4_x_grid"]
    fig4, axes4 = plt.subplots(3, 5, figsize=(11.6, 6.9))
    plt.subplots_adjust(hspace=0.40, wspace=0.22)
    for idx_m, mk in enumerate(mk_names_14):
        ax_m = axes4[idx_m // 5, idx_m % 5]
        style_precision_ax(ax_m, cat_y=True, grid="x")
        vals = s_diag[f"s4_vals_{idx_m}"]
        c0 = s_diag[f"s4_c0_{idx_m}"]
        c1 = s_diag[f"s4_c1_{idx_m}"]
        tau = float(s_diag[f"s4_tau_{idx_m}"][0])
        q75 = float(np.percentile(vals, 75.0))
        ax_m.hist(vals[vals > 0.02], bins=24, density=True, color="#CBD5E1", edgecolor=INK, lw=0.38, zorder=2)
        ax_m.plot(xg_14, c0, color="#1D4ED8", lw=1.35, label="$\\mathcal{N}_-$", zorder=3)
        ax_m.plot(xg_14, c1, color="#B91C1C", lw=1.45, label="$\\mathcal{N}_+$", zorder=3)
        ax_m.axvline(q75, color="#334155", ls=":", lw=1.20, label="q75", zorder=4)
        ax_m.axvline(tau, color="#B91C1C", ls="-", lw=1.40, label="$\\tau_p^{\\star}$", zorder=4)
        pos_pct = 100.0 * float((vals >= tau).mean())
        ax_m.set_xlim(0, max(float(np.percentile(vals, 99.5)) * 1.12, 1.8))
        ax_m.set_yticks([])
        ax_m.set_title(f"{mk} ({pos_pct:.0f}%)", fontsize=9.3)
        if idx_m >= 9:
            ax_m.set_xlabel("Measured CLR", fontsize=9.2)
        plabel(ax_m, chr(ord("a") + idx_m), x=-0.08, y=1.05)
    ax_leg4 = axes4[2, 4]
    ax_leg4.axis("off")
    leg_h4 = [
        Line2D([0], [0], color="#1D4ED8", lw=1.6, label="Negative peak ($\\mathcal{N}_-$)"),
        Line2D([0], [0], color="#B91C1C", lw=1.6, label="Positive peak ($\\mathcal{N}_+$)"),
        Line2D([0], [0], color="#334155", lw=1.4, ls=":", label="Fixed 75th-pct (q75)"),
        Line2D([0], [0], color="#B91C1C", lw=1.6, ls="-", label="2-GMM Bayes $\\tau_p^{\\star}$"),
    ]
    ax_leg4.legend(handles=leg_h4, frameon=True, facecolor="white", edgecolor=INK, fontsize=9.3, loc="center", title="1D Gate Legend", title_fontsize=9.3)
    save_fig(fig4, SUPP_FIG_DIR, "FigureS4")

    # --- Figure S5 (a-d): Complete 18-Sample GSE166489 Single-Cell QC & TCR Coverage Atlas ---
    df_qc = pd.read_csv(TABLES_DIR / "v42_gse166489_18sample_qc.csv")
    fig5, axes5 = plt.subplots(4, 1, figsize=(12.25, 7.79), sharex=True)
    plt.subplots_adjust(hspace=0.14)
    cond_col_map = {"Adult_Healthy": "#475569", "Pediatric_Healthy": "#1D4ED8", "MIS-C_Recovered": "#047857", "MIS-C_Moderate": "#B45309", "MIS-C_Severe": "#B91C1C"}
    s_list = df_qc["sample_id"].tolist()
    xs5 = np.arange(len(s_list))
    bcols5 = [cond_col_map[str(c)] for c in df_qc["condition"]]

    style_precision_ax(axes5[0], cat_x=True, grid="y")
    axes5[0].axhline(float((df_qc["n_cells"] / 1000.0).median()), color="#475569", ls=":", lw=0.75, zorder=2)
    axes5[0].bar(xs5 - 0.18, df_qc["n_cells"] / 1000.0, width=0.35, color=bcols5, edgecolor=INK, lw=0.78, label="QC PBMC cells ($\\times 10^3$)", zorder=3)
    ax0_t = axes5[0].twinx()
    ax0_t.spines["right"].set_visible(True); ax0_t.spines["right"].set_edgecolor(INK); ax0_t.spines["right"].set_linewidth(0.85)
    ax0_t.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax0_t.tick_params(axis="y", which="both", direction="in", colors="#5B21B6")
    ax0_t.plot(xs5 + 0.18, df_qc["tcr_pct"], color="#5B21B6", marker="o", markerfacecolor="white", markeredgewidth=0.85, lw=1.5, markersize=4.6, label="Paired $\\alpha\\beta$ TCR (%)")
    ax0_t.set_ylabel("Matched TCR (%)", fontsize=8.7, color="#5B21B6", fontweight="bold")
    ax0_t.set_ylim(0, 88)
    axes5[0].set_ylabel("QC cells ($\\times 10^3$)")
    axes5[0].set_title("Single-cell quality control, library depth, and matched TCR coverage across all 18 GSE166489 draws (162,777 cells)")
    plabel(axes5[0], "a", x=-0.05)

    for row_i, (k_suf, ylbl_s, ltr_s, hline_v) in enumerate([
        ("log10_umi", "$\\log_{10}$(RNA UMIs)", "b", None),
        ("genes", "Detected genes", "c", 200.0),
        ("mt_pct", "Mitochondrial %", "d", 20.0),
    ], start=1):
        ax_q = axes5[row_i]
        style_precision_ax(ax_q, cat_x=True, grid="y")
        data_v = [s_diag[f"s5_{sid}_{k_suf}"] for sid in s_list]
        all_med = float(np.median(np.concatenate(data_v)))
        ax_q.axhline(all_med, color="#1D4ED8", ls=":", lw=0.75, zorder=2)
        parts = ax_q.violinplot(data_v, positions=xs5, widths=0.68, showmeans=False, showextrema=False, showmedians=True)
        for pc, bc in zip(parts["bodies"], bcols5):
            pc.set_facecolor(bc); pc.set_edgecolor(INK); pc.set_linewidth(0.65); pc.set_alpha(0.88)
        parts["cmedians"].set_edgecolor(INK); parts["cmedians"].set_linewidth(1.3)
        if hline_v is not None:
            ax_q.axhline(hline_v, color="#B91C1C", ls="--", lw=0.95, zorder=4)
        ax_q.set_ylabel(ylbl_s)
        plabel(ax_q, ltr_s, x=-0.05)

    axes5[3].set_xticks(xs5)
    axes5[3].set_xticklabels(s_list, rotation=28, ha="right", fontweight="bold", fontsize=8.5)
    axes5[3].set_xlabel("GSE166489 donor / longitudinal clinical draw")
    save_fig(fig5, SUPP_FIG_DIR, "FigureS5")

    # --- Figure S6 (a-d): Additional Cis-Chromatin Loci (CD8A & SIRPA) + RTX 5070 Stage Latency & Batch VRAM Scaling ---
    df_trk = pd.read_csv(TABLES_DIR / "v41_teaseq_cis_genomic_tracks.csv")
    df_lat = pd.read_csv(TABLES_DIR / "v42_laptop_latency_vram_scaling.csv")
    fig6, axes6 = plt.subplots(2, 2, figsize=(11.55, 6.28))
    plt.subplots_adjust(hspace=0.32, wspace=0.26)

    style_precision_ax(axes6[0, 0], grid="both")
    cd8_t = df_trk[df_trk["locus"] == "CD8A (CD8a)"].sort_values("mid_kb")
    axes6[0, 0].bar(cd8_t["mid_kb"] - 1.2, cd8_t["open_pct_CD8_EffMem_KLRG1"], width=2.2, color="#047857", edgecolor=INK, lw=0.70, label="CD8 Eff/Mem", zorder=3)
    axes6[0, 0].bar(cd8_t["mid_kb"] + 1.2, cd8_t["open_pct_Myeloid_Mono_cDC"], width=2.2, color="#B45309", edgecolor=INK, lw=0.70, label="Myeloid", zorder=3)
    axes6[0, 0].axvline(86808.4, color="#B91C1C", ls="--", lw=1.10, label="CD8A TSS", zorder=4)
    axes6[0, 0].xaxis.set_major_locator(plt.MaxNLocator(4))
    axes6[0, 0].set_xlabel("chr2 coordinate (kb, GRCh38)"); axes6[0, 0].set_ylabel("Accessible cells (%)")
    axes6[0, 0].set_title("CD8A cis-chromatin accessibility (X061_W1)")
    axes6[0, 0].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.3)
    plabel(axes6[0, 0], "a")

    style_precision_ax(axes6[0, 1], grid="both")
    sir_t = df_trk[df_trk["locus"] == "SIRPA (CD172a)"].sort_values("mid_kb")
    axes6[0, 1].bar(sir_t["mid_kb"] - 0.35, sir_t["open_pct_Myeloid_Mono_cDC"], width=0.65, color="#B45309", edgecolor=INK, lw=0.70, label="Myeloid", zorder=3)
    axes6[0, 1].bar(sir_t["mid_kb"] + 0.35, sir_t["open_pct_CD4_Naive_RA"], width=0.65, color="#1D4ED8", edgecolor=INK, lw=0.70, label="CD4 Naive", zorder=3)
    axes6[0, 1].axvline(1894.17, color="#B91C1C", ls="--", lw=1.10, label="SIRPA TSS", zorder=4)
    axes6[0, 1].set_xlabel("chr20 coordinate (kb, GRCh38)"); axes6[0, 1].set_ylabel("Accessible cells (%)")
    axes6[0, 1].set_title("SIRPA (CD172a) cis-chromatin accessibility")
    axes6[0, 1].legend(frameon=True, facecolor="white", edgecolor=INK, fontsize=7.3)
    plabel(axes6[0, 1], "b")

    style_precision_ax(axes6[1, 0], cat_y=True, grid="x")
    stg = df_lat[df_lat["kind"] == "stage_breakdown"].reset_index(drop=True)
    ys6 = np.arange(len(stg))
    axes6[1, 0].axvline(1.0, color="#64748B", ls=":", lw=0.75, zorder=2)
    axes6[1, 0].barh(ys6, stg["sec"], height=0.56, color=["#475569", "#B91C1C", "#1D4ED8", "#047857", "#B45309"], edgecolor=INK, lw=0.80, zorder=3)
    for ri, r in stg.iterrows():
        axes6[1, 0].text(float(r["sec"]) + 0.08, int(ri), f"{r['sec']:.2f} s", va="center", fontsize=7.8, fontweight="bold")
    axes6[1, 0].set_yticks(ys6); axes6[1, 0].set_yticklabels(stg["workload"], fontweight="bold", fontsize=8.0)
    axes6[1, 0].set_xlim(0, float(stg["sec"].max()) * 1.28)
    axes6[1, 0].set_xlabel("Wall-clock latency (s, n_ref=500 -> 6,941 query cells)")
    axes6[1, 0].set_title("End-to-end stage latency breakdown (RTX 5070)")
    plabel(axes6[1, 0], "c")

    bs_df = df_lat[df_lat["kind"] == "batch_scaling"]
    ax_bs = axes6[1, 1]
    ax_bs.set_xscale("log", base=2)
    full_box(ax_bs, 0.85)
    ax_bs.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_bs.yaxis.grid(True, which="major", ls="--", lw=0.45, color="#CBD5E1", alpha=0.85)
    ax_bs.xaxis.grid(True, which="major", ls="--", lw=0.45, color="#CBD5E1", alpha=0.85)
    ax_vr = ax_bs.twinx()
    ax_vr.spines["right"].set_visible(True); ax_vr.spines["right"].set_edgecolor(INK); ax_vr.spines["right"].set_linewidth(0.85)
    ax_vr.yaxis.set_minor_locator(AutoMinorLocator(2))
    h_bs = []
    for w_sub, col_w, lbl_w in [
        ("GSE194122_RNA", "#1D4ED8", "RNA (512 tok)"),
        ("GSE158013_RNA+ATAC", "#B91C1C", "RNA+ATAC (1536 tok)"),
    ]:
        sw = bs_df[bs_df["workload"].str.startswith(w_sub)].sort_values("batch_size")
        l1, = ax_bs.plot(sw["batch_size"], sw["cells_per_sec"], marker="o", lw=1.95, color=col_w, markeredgecolor=INK, markeredgewidth=0.65, label=f"{lbl_w} speed")
        l2, = ax_vr.plot(sw["batch_size"], sw["peak_vram_mb"] / 1024.0, marker="s", ls="--", lw=1.40, color=col_w, markeredgecolor=INK, markeredgewidth=0.55, alpha=0.88, label=f"{lbl_w} VRAM")
        h_bs.extend([l1, l2])
    ax_vr.axhline(8.0, color="#B91C1C", ls="--", lw=1.05)
    ax_vr.text(980.0, 7.35, "8.0 GB VRAM ceiling", fontsize=7.1, fontweight="bold", color="#B91C1C", ha="right")
    ax_bs.set_ylim(0, 2950)
    ax_bs.set_xlabel("Inference batch size ($\\log_2$ scale)")
    ax_bs.set_ylabel("Throughput (cells/s, solid)")
    ax_vr.set_ylabel("Peak GPU VRAM (GB, dashed)"); ax_vr.set_ylim(0, 10.6)
    ax_bs.set_title("Batch-size throughput & 8 GB VRAM scaling")
    ax_bs.legend(handles=h_bs, frameon=True, facecolor="white", edgecolor=INK, ncol=2, fontsize=6.8, loc="upper left")
    plabel(ax_bs, "d")
    save_fig(fig6, SUPP_FIG_DIR, "FigureS6")


def main() -> None:
    make_figure2_mechanisms()
    make_figure3_crossover_scorecard()
    make_figure4_hygiene_gmm_manifold()
    make_figure5_misc_18donor_atlas()
    make_figure6_tcr_superantigen_and_uncertainty()
    make_figure7_teaseq_triomic_and_chromatin()
    make_figure8_foundation_models()
    make_supplementary_figures()
    print("ALL FLAGSHIP MAIN (2-8) & SUPPLEMENTARY (S1-S6) FIGURES GENERATED!", flush=True)


if __name__ == "__main__":
    main()
