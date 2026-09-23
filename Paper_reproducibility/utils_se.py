import re

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.multitest import multipletests

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

#: Etiology map used by the TCGA bubble plot.
ETIOLOGY_TCGA = {
    # Unknown
    "SBS12":  "Unknown",    "SBS23":  "Unknown",    "SBS24":  "Unknown",
    "SBS33":  "Unknown",    "SBS36":  "Unknown",    "SBS37":  "Unknown",
    "SBS39":  "Unknown",    "SBS41":  "Unknown",    "SBS88":  "Unknown",
    "SBS89":  "Unknown",    "SBS90":  "Unknown",    "SBS93":  "Unknown",
    # MMR
    "SBS6":   "MMR",        "SBS14":  "MMR",        "SBS15":  "MMR",
    "SBS20":  "MMR",        "SBS21":  "MMR",        "SBS26":  "MMR",
    "SBS44":  "MMR",
    # Clock-like
    "SBS1":   "Clock-like", "SBS5":   "Clock-like", "SBS40a": "Clock-like",
    "SBS40b": "Clock-like", "SBS40c": "Clock-like",
    # UV
    "SBS7a":  "UV",         "SBS7b":  "UV",         "SBS7c":  "UV",
    "SBS7d":  "UV",         "SBS38":  "UV",
    # POL def
    "SBS10a": "POL def",    "SBS10b": "POL def",    "SBS10c": "POL def",
    "SBS10d": "POL def",    "SBS28":  "POL def",
    # Treatment
    "SBS11":  "Treatment",  "SBS25":  "Treatment",  "SBS31":  "Treatment",
    "SBS32":  "Treatment",  "SBS35":  "Treatment",
    # Tobacco
    "SBS4":   "Tobacco",    "SBS29":  "Tobacco",    "SBS92":  "Tobacco",
    # ROS
    "SBS17a": "ROS",        "SBS17b": "ROS",        "SBS18":  "ROS",
    # AID
    "SBS9":   "AID",        "SBS84":  "AID",        "SBS85":  "AID",
    # APOBEC
    "SBS2":   "APOBEC",     "SBS13":  "APOBEC",
    # others
    "SBS3":   "HRD",        "SBS8":   "HRD/NER",    "SBS30":  "BER def",
    "SBS42":  "Haloalkane", "SBS22a": "Aristolochic acid",
}

#: Etiology map used by the CPTAC bubble plot. It differs from the TCGA one
#: (SBS40 instead of SBS40a/b/c, no SBS12), exactly as in the notebook.
ETIOLOGY_CPTAC = {
    "SBS1":  "Clock-like", "SBS5":  "Clock-like", "SBS40": "Clock-like",
    "SBS2":  "APOBEC",     "SBS13": "APOBEC",
    "SBS3":  "HRD",
    "SBS4":  "Tobacco",    "SBS29": "Tobacco",    "SBS92": "Tobacco",
    "SBS6":  "MMR",        "SBS14": "MMR",        "SBS15": "MMR",
    "SBS20": "MMR",        "SBS21": "MMR",        "SBS26": "MMR",
    "SBS44": "MMR",
    "SBS7a": "UV",         "SBS7b": "UV",         "SBS7c": "UV",
    "SBS7d": "UV",         "SBS38": "UV",
    "SBS8":  "HRD/NER",
    "SBS17a":"ROS",        "SBS17b":"ROS",        "SBS18": "ROS",
    "SBS9":  "AID",        "SBS84": "AID",        "SBS85": "AID",
    "SBS10a":"POL def",    "SBS10b":"POL def",    "SBS10c":"POL def",
    "SBS10d":"POL def",    "SBS28": "POL def",
    "SBS11": "Treatment",  "SBS25": "Treatment",  "SBS31": "Treatment",
    "SBS32": "Treatment",  "SBS35": "Treatment",
    "SBS22a":"Aristolochic acid",
    "SBS30": "BER def",    "SBS42": "Haloalkane",
    "SBS23": "Unknown",    "SBS24": "Unknown",    "SBS33": "Unknown",
    "SBS36": "Unknown",    "SBS37": "Unknown",    "SBS39": "Unknown",
    "SBS41": "Unknown",    "SBS88": "Unknown",    "SBS89": "Unknown",
    "SBS90": "Unknown",    "SBS93": "Unknown",
}

#: CPTAC short code -> TCGA project id.
CPTAC_TO_TCGA = {
    "LSCC": "TCGA-LUSC", "PDA": "TCGA-PAAD", "CCRCC": "TCGA-KIRC",
    "LUAD": "TCGA-LUAD", "GBM": "TCGA-GBM",
}

#: TCGA project id -> short label (cosine-similarity figure).
TCGA_TO_SHORT = {
    "TCGA-LUAD": "LUAD", "TCGA-PAAD": "PAAD",
    "TCGA-KIRC": "KIRC", "TCGA-LUSC": "LUSC", "TCGA-GBM": "GBM",
}

#: TCGA project id -> CPTAC short code (Pearson bubble figure).
TCGA_TO_CPTAC_SHORT = {
    "TCGA-LUSC": "LSCC", "TCGA-PAAD": "PDA", "TCGA-KIRC": "CCRCC",
    "TCGA-LUAD": "LUAD", "TCGA-GBM": "GBM",
}

SELECTED_PROJECTS = ['TCGA-GBM', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-KIRC', 'TCGA-PAAD']

PROJECT_LABELS = {
    'TCGA-GBM':  'GBM',
    'TCGA-LUAD': 'LUAD',
    'TCGA-LUSC': 'LUSC',
    'TCGA-KIRC': 'KIRC',
    'TCGA-PAAD': 'PAAD',
}

COHORT_PALETTE = {'TCGA': '#3B6FB6', 'CPTAC': '#E07A2B'}


def _viridis_r():
    """Deferred lookup so importing the module never touches the colormap API."""
    return plt.cm.get_cmap("viridis_r")


def model_colors():
    """(violet, blue) = (Hist2Sig, RF) as used throughout the notebook."""
    viridis_r = _viridis_r()
    return (matplotlib.colors.to_hex(viridis_r(0.92)),
            matplotlib.colors.to_hex(viridis_r(0.65)))


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

def get_etiology(sbs, etiology=None):
    etiology = ETIOLOGY_TCGA if etiology is None else etiology
    return etiology.get(sbs, "Unknown")


def make_ylabel(sbs, etiology=None):
    etio = get_etiology(sbs, etiology)
    if etio != "Unknown":
        return f"({etio}) {sbs}"
    return sbs


def sbs_sort_key(s):
    s_clean = s.replace("SBS", "")
    num_part = "".join(c for c in s_clean if c.isdigit())
    suf_part = "".join(c for c in s_clean if not c.isdigit())
    try:
        return (int(num_part), suf_part)
    except:
        return (9999, s_clean)


def sig_sort_key(s):
    """Numeric-only sort key, used by the Pearson bubble figures."""
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _sig_sort_key(s):
    """Digit-concatenating sort key, used by the composition figure."""
    digits = ''.join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def sig_mark(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def significance_mark(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def mean_sem(x):
    x = np.array(x)
    mean = np.mean(x)
    sem = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return mean, sem


def normalize_by_sample(true_vals, pred_vals, sigs):
    t = true_vals[sigs].copy().astype(float)
    p = pred_vals[sigs].copy().astype(float)
    t = t.div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return t, p


def get_signature_columns(df):
    return [c for c in df.columns if c.startswith('SBS')]


# ════════════════════════════════════════════════════════════════════════════
# 1. Loading
# ════════════════════════════════════════════════════════════════════════════

def load_fold_predictions(dir_files):
    """Concatenate the 5 validation-fold prediction files (notebook cell 5)."""
    dfs = []
    for i in range(1, 6):
        df = pd.read_csv(f'./{dir_files}/fold{i}_val_predictions_activities.csv')
        df['fold'] = i
        dfs.append(df)
    return pd.concat(dfs)


def split_true_pred(all_data):
    """Split the concatenated folds into true / predicted frames (cell 6)."""
    true_activities = all_data.loc[:, ~all_data.columns.str.startswith("pred_")].iloc[:, :-1]
    true_activities.columns = true_activities.columns.str.replace("^true_", "", regex=True)

    pred_activities = all_data.loc[:, ~all_data.columns.str.startswith("true_")].iloc[:, :-1]
    pred_activities.columns = pred_activities.columns.str.replace("^pred_", "", regex=True)

    return true_activities, pred_activities


# ════════════════════════════════════════════════════════════════════════════
# 2. TCGA signature landscape (bubble plot)
# ════════════════════════════════════════════════════════════════════════════

def plot_bubble_tcga(true_activities, save_dir=None):
    """Signature landscape across TCGA tumour types (cell 7)."""
    df = true_activities.copy()
    sbs_cols = [col for col in df.columns if col.startswith("SBS")]

    # normalize row-wise
    df_norm = df.copy()
    row_sums = df[sbs_cols].sum(axis=1)
    df_norm[sbs_cols] = df[sbs_cols].div(row_sums, axis=0).fillna(0)

    results = []
    for tumour, group in df.groupby("Project ID"):
        n = len(group)
        group_norm = df_norm.loc[group.index]
        for sbs in sbs_cols:
            mean_prop   = group_norm[sbs].mean()
            pct_present = (group[sbs] > 0).sum() / n * 100
            results.append({
                "Tumour":     tumour,
                "SBS":        sbs,
                "MeanProp":   mean_prop,
                "PctPresent": pct_present,
            })

    bubble_df   = pd.DataFrame(results)
    pivot_size  = bubble_df.pivot(index="SBS", columns="Tumour", values="MeanProp")
    pivot_color = bubble_df.pivot(index="SBS", columns="Tumour", values="PctPresent")

    # sort columns by decreasing sample count
    tumour_counts = df.groupby("Project ID").size().sort_values(ascending=False)
    pivot_size    = pivot_size[tumour_counts.index]
    pivot_color   = pivot_color[tumour_counts.index]

    # sort rows by SBS number
    sorted_sbs  = sorted(pivot_size.index, key=sbs_sort_key)
    pivot_size  = pivot_size.loc[sorted_sbs]
    pivot_color = pivot_color.loc[sorted_sbs]

    # drop signatures that are never present
    keep        = (pivot_size > 0).any(axis=1)
    pivot_size  = pivot_size.loc[keep]
    pivot_color = pivot_color.loc[keep]

    signatures = list(pivot_size.index)
    tumours    = list(pivot_size.columns)
    n_t = len(tumours)
    n_s = len(signatures)

    fig_w = max(18, n_t * 0.55)
    fig_h = max(10, n_s * 0.42) + 2.5
    fig   = plt.figure(figsize=(fig_w, fig_h))

    left   = 0.13
    right  = 0.80
    bottom = 0.05

    header_inch = 1.8
    header_frac = header_inch / fig_h
    gap_frac    = 0.005
    top_plot    = 1.0 - header_frac - gap_frac
    plot_h      = top_plot - bottom

    ax = fig.add_axes([left, bottom, right - left, plot_h])

    cmap       = cm.viridis
    norm_color = mcolors.Normalize(vmin=0, vmax=100)
    max_size   = 500

    for i, sig in enumerate(signatures):
        for j, tumour in enumerate(tumours):
            prop = pivot_size.loc[sig, tumour]
            pct  = pivot_color.loc[sig, tumour]
            if prop > 0:
                ax.scatter(j, i,
                           s=prop * max_size,
                           color=cmap(norm_color(pct)),
                           edgecolors="none", zorder=3)

    ax.set_xticks(range(n_t))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False)

    ylabels = [make_ylabel(s, ETIOLOGY_TCGA) for s in signatures]
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_ylabel("Mutational signature", fontsize=15)

    ax.set_xlim(-0.5, n_t - 0.5)
    ax.set_ylim(-0.5, n_s - 0.5)
    ax.grid(True, color="lightgray", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # header: sample counts + tumour names
    ax_top = fig.add_axes([left, top_plot + gap_frac, right - left, header_frac])
    ax_top.set_xlim(-0.5, n_t - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    circle_y = 0.10
    name_y   = 0.28

    for j, tumour in enumerate(tumours):
        n = tumour_counts[tumour]
        ax_top.text(j, circle_y, str(n),
                    ha="center", va="center",
                    fontsize=8, color="navy",
                    bbox=dict(boxstyle="circle,pad=0.35",
                              fc="lightblue", ec="steelblue", lw=1))
        ax_top.text(j, name_y, tumour,
                    ha="left", va="bottom",
                    fontsize=9, rotation=40, color="black")

    cbar_left   = right + 0.02
    cbar_bottom = bottom + 0.25
    cbar_h      = plot_h * 0.45
    ax_cbar     = fig.add_axes([cbar_left, cbar_bottom, 0.02, cbar_h])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("% of samples with signature", fontsize=11,
                   rotation=90, labelpad=15)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.tick_params(labelsize=10)

    legend_props   = [0.25, 0.50, 0.75, 1.0]
    legend_handles = [
        plt.scatter([], [], s=p * max_size, color="gray",
                    edgecolors="none", label=f"{p:.2f}")
        for p in legend_props
    ]
    legend_bottom = cbar_bottom + cbar_h + 0.04
    ax_leg = fig.add_axes([cbar_left - 0.01, legend_bottom, 0.10, 0.20])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=legend_handles,
        title="Normalized\nexposure",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        scatterpoints=1,
        borderpad=0.8,
        labelspacing=0.8,
    )

    if save_dir is not None:
        plt.savefig(f"{save_dir}/bubble_sbs_tumour.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 3. Cosine similarity per tumour type (TCGA)
# ════════════════════════════════════════════════════════════════════════════

def plot_cosine_tcga(true_activities, pred_f_activities, save_dir=None):
    """Cosine-similarity boxplot per tumour type (cell 9).

    Returns ``(df1, df2, df_cos)``; ``df1`` is reused by later cells.
    """
    df1 = true_activities.iloc[:, 6:]
    df2 = pred_f_activities.iloc[:, 6:][df1.columns]
    cos_sim = cosine_similarity(df1.values, df2.values).diagonal()
    df_cos = df1.copy()
    df_cos["cosine_similarity"] = cos_sim
    df_cos["Project ID"] = true_activities["Project ID"].values

    mean_order = (
        df_cos
        .groupby("Project ID")["cosine_similarity"]
        .mean()
        .sort_values(ascending=False)
    )
    order = mean_order.index

    counts = df_cos["Project ID"].value_counts()
    labels = [f"{proj}\n(n={counts[proj]})" for proj in order]

    rng = np.random.default_rng(0)
    n_random = 10000
    dim = 28
    N = df1.max().max()
    x = rng.uniform(0, N, size=(n_random, dim))
    y = rng.uniform(0, N, size=(n_random, dim))
    random_cosines = np.sum(x * y, axis=1) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )
    random_mean = random_cosines.mean()

    plt.rcParams.update({
        "font.size":        22,
        "axes.titlesize":   22,
        "axes.labelsize":   22,
        "xtick.labelsize":  22,
        "ytick.labelsize":  22,
        "legend.fontsize":  22,
    })

    fig, ax = plt.subplots(figsize=(24, 10))

    sns.boxplot(
        data=df_cos,
        x="Project ID",
        y="cosine_similarity",
        order=order,
        palette="viridis",
        showfliers=False,
        ax=ax,
    )

    ax.set_xlabel("Tumor type", labelpad=10)
    ax.set_ylabel("Cosine similarity", labelpad=10)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, ha="center")

    ax.tick_params(axis="both", which="major", length=6, width=1.2)

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(
            f"{save_dir}/cosine_similarity_per_tumor_boxplot_with_random_baseline.pdf",
            dpi=150, bbox_inches="tight")

    plt.show()

    return df1, df2, df_cos


# ════════════════════════════════════════════════════════════════════════════
# 4. Pearson bubble plot (TCGA)
# ════════════════════════════════════════════════════════════════════════════

def plot_pearson_bubble_tcga(true_activities, pred_f_activities,
                             r_threshold=0.2, min_n=10, min_n_tumor=10,
                             save_dir=None):
    """Per-tumour, per-signature Pearson r between true and predicted (cell 10)."""
    BIN_EDGES  = [0.2, 0.3, 0.5, 0.7, 1.01]
    BIN_LABELS = ["0.2 – 0.3", "0.3 – 0.5", "0.5 – 0.7", "> 0.7"]

    viridis_r   = _viridis_r()
    BIN_COLORS  = [matplotlib.colors.to_hex(viridis_r(v)) for v in [0.10, 0.35, 0.65, 0.92]]
    BUBBLE_SIZE = 700

    df_true = true_activities.iloc[:, 6:]
    df_pred = pred_f_activities.iloc[:, 6:][df_true.columns]
    all_sigs    = [c for c in df_true.columns if c.startswith("SBS")]
    project_ids = true_activities["Project ID"]
    tumors      = project_ids.unique()
    tumors_to_plot = [t for t in tumors if (project_ids == t).sum() >= min_n_tumor]

    records = []
    for tumor in tumors_to_plot:
        idx      = np.where((project_ids == tumor).values)[0]
        true_raw = df_true.iloc[idx][all_sigs].reset_index(drop=True)
        pred_raw = df_pred.iloc[idx][all_sigs].reset_index(drop=True)
        t_norm, p_norm = normalize_by_sample(true_raw, pred_raw, all_sigs)

        for sig in all_sigs:
            t_v, p_v = t_norm[sig], p_norm[sig]
            n = len(t_v)
            if n >= min_n and t_v.std() > 0 and p_v.std() > 0:
                r, pval = pearsonr(t_v, p_v)
                records.append(dict(tumor=tumor, sig=sig, r=r, pval=pval, n=n))

    df = pd.DataFrame(records)

    _, df["pval_fdr"], _, _ = multipletests(df["pval"], method="fdr_bh")

    df_sig = df[df["r"] >= r_threshold].copy()

    all_present_sigs = df_sig["sig"].unique().tolist()
    sig_order = sorted(all_present_sigs, key=sig_sort_key, reverse=False)

    tumor_order = (
        df_sig.groupby("tumor")["sig"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    r_pivot    = df_sig.pivot_table(index="sig", columns="tumor", values="r",        aggfunc="first").reindex(index=sig_order, columns=tumor_order)
    pval_pivot = df_sig.pivot_table(index="sig", columns="tumor", values="pval_fdr", aggfunc="first").reindex(index=sig_order, columns=tumor_order)

    n_sigs   = len(sig_order)
    n_tumors = len(tumor_order)

    fig_w = max(8, n_tumors * 0.65 + 2.5)
    fig_h = max(5, n_sigs   * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    def r_to_bin(r_val):
        for k in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[k] <= r_val < BIN_EDGES[k + 1]:
                return k
        return len(BIN_COLORS) - 1

    for i, sig in enumerate(sig_order):
        for j, tumor in enumerate(tumor_order):
            r_val = r_pivot.loc[sig, tumor]    if tumor in r_pivot.columns    else np.nan
            p_val = pval_pivot.loc[sig, tumor] if tumor in pval_pivot.columns else np.nan
            if np.isnan(r_val):
                continue
            bin_idx = r_to_bin(r_val)
            color   = BIN_COLORS[bin_idx]
            ax.scatter(j, i, s=BUBBLE_SIZE, color=color,
                       edgecolors="white", linewidths=0.5, zorder=3)

            if not np.isnan(p_val) and p_val < 0.05:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold", zorder=4)

    ax.set_xticks(range(n_tumors))
    ax.set_xticklabels(tumor_order, rotation=45, ha="right", fontsize=18)
    ax.set_yticks(range(n_sigs))
    ax.set_yticklabels(sig_order, fontsize=18)
    ax.set_xlim(-0.7, n_tumors - 0.3)
    ax.set_ylim(-0.7, n_sigs   - 0.3)
    ax.set_facecolor("#F7F7F7")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="white", linewidth=0.8, zorder=0)

    legend_handles = [
        plt.scatter([], [], s=BUBBLE_SIZE, color=BIN_COLORS[k],
                    edgecolors="white", linewidths=0.5, label=BIN_LABELS[k])
        for k in range(len(BIN_LABELS))
    ]
    ax.legend(handles=legend_handles, title="Pearson r", loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=18, title_fontsize=18,
              frameon=False, scatterpoints=1)

    ax.set_xlabel("Tumor type", fontsize=18)
    ax.set_ylabel("Signature", fontsize=18)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/bubble_pearson_tumor_sig.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    return df


# ════════════════════════════════════════════════════════════════════════════
# 5. Top-3 metrics (TCGA)
# ════════════════════════════════════════════════════════════════════════════

def top3_overlap_score(true_row, pred_row):
    true_top3 = set(np.argsort(true_row)[-3:])
    pred_top3 = set(np.argsort(pred_row)[-3:])
    return len(true_top3 & pred_top3) / 3.0


def top3_exact_position_score(true_row, pred_row):
    true_top3 = np.argsort(true_row)[-3:][::-1]  # descending: 1st, 2nd, 3rd
    pred_top3 = np.argsort(pred_row)[-3:][::-1]
    matches = sum(t == p for t, p in zip(true_top3, pred_top3))
    return matches / 3.0


def compute_top3_metrics(true_activities, pred_f_activities, df1):
    """Sample-level top-3 overlap / exact-position scores (cell 11).

    Returns ``(df_true, df_pred, df_metrics, metrics_by_project)``.
    """
    df_true = true_activities.iloc[:, 6:]
    df_pred = pred_f_activities.iloc[:, 6:][df1.columns]
    project_id = true_activities["Project ID"].values

    assert df_true.shape == df_pred.shape
    assert (df_true.columns == df_pred.columns).all()

    overlap_scores = [top3_overlap_score(df_true.iloc[i].values, df_pred.iloc[i].values)
                      for i in range(len(df_true))]
    exact_scores   = [top3_exact_position_score(df_true.iloc[i].values, df_pred.iloc[i].values)
                      for i in range(len(df_true))]

    df_metrics = pd.DataFrame({
        "Project ID":   project_id,
        "Top3_Overlap": overlap_scores,
        "Top3_Exact":   exact_scores
    })

    metrics_by_project = (df_metrics.groupby("Project ID")[["Top3_Overlap", "Top3_Exact"]]
        .agg(["mean", "median", "count"])
        .sort_values(("Top3_Overlap", "mean"), ascending=False))

    return df_true, df_pred, df_metrics, metrics_by_project


# ════════════════════════════════════════════════════════════════════════════
# 6. RF baseline trained on tumour type
# ════════════════════════════════════════════════════════════════════════════

def train_rf_tumor_types(dir_files, n_estimators=1000, random_state=42, n_jobs=-1):
    """Per-fold RF on one-hot tumour type + a final model (cell 13).

    Returns a dict with ``df_final``, ``y_cols``, ``encoder`` (last fold),
    ``final_model``, ``X_train_all``, ``y_train_all`` — the same objects the
    notebook left in the namespace.
    """
    all_results = []
    all_X_train = []
    all_y_train = []

    for i in range(1, 6):
        train_path = f"./{dir_files}/fold{i}_train_predictions_activities.csv"
        val_path = f"./{dir_files}/fold{i}_val_predictions_activities.csv"

        print(f"Processing fold {i}...")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        y_cols = [c for c in train_df.columns if c.startswith("true_SBS")]

        y_train = train_df[y_cols]
        y_val = val_df[y_cols]

        encoder = OneHotEncoder(handle_unknown="ignore")
        X_train = encoder.fit_transform(train_df[["Project ID"]])
        X_val = encoder.transform(val_df[["Project ID"]])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        pred_cols = [c.replace("true_", "rf_pred_") for c in y_cols]
        pred_df = pd.DataFrame(y_pred, columns=pred_cols)

        meta_cols = [
            c for c in ["Case ID", "Project ID", "Sample ID", "File ID"]
            if c in val_df.columns
        ]

        fold_df = pd.concat([
            val_df[meta_cols].reset_index(drop=True),
            val_df[y_cols].reset_index(drop=True),
            pred_df.reset_index(drop=True)
        ], axis=1)

        fold_df["fold"] = i

        all_results.append(fold_df)
        all_X_train.append(X_val)
        all_y_train.append(y_val)

    df_final = pd.concat(all_results, ignore_index=True)

    X_train_all = pd.concat(
        [pd.DataFrame(x.toarray()) if hasattr(x, 'toarray') else pd.DataFrame(x)
         for x in all_X_train],
        ignore_index=True
    )

    y_train_all = pd.concat(all_y_train, ignore_index=True)

    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )

    final_model.fit(X_train_all, y_train_all)

    return {
        "df_final":    df_final,
        "y_cols":      y_cols,
        "encoder":     encoder,
        "final_model": final_model,
        "X_train_all": X_train_all,
        "y_train_all": y_train_all,
    }


# ════════════════════════════════════════════════════════════════════════════
# 7. Top-3 overlap + signature diversity (TCGA)
# ════════════════════════════════════════════════════════════════════════════

def top3_figures_tcga(df_true, df_pred, df_final, df_metrics, save_dir=None):
    """Top-3 overlap histograms and signature diversity per tumour (cell 14).

    Returns a dict with the objects later cells need (``nn_pred``, ``nn_true``,
    ``rf_pred``, ``projects``, ``unique_projects``, ``all_*``).
    """
    V_VIOLET, V_BLUE = model_colors()
    color_nn = V_VIOLET
    color_rf = V_BLUE

    nn_pred = df_pred.copy()
    nn_true = df_true.copy()

    rf_pred_cols = [c for c in df_final.columns if c.startswith("rf_pred_SBS")]
    rf_pred = df_final[rf_pred_cols].copy()
    rf_pred.columns = [c.replace("rf_pred_", "") for c in rf_pred.columns]

    projects        = df_metrics["Project ID"]
    unique_projects = projects.unique()

    categories      = [0, 1/3, 2/3, 1]
    category_labels = ["0", "1/3", "2/3", "1"]

    bar_width = 0.12
    offset_nn = -bar_width / 2
    offset_rf =  bar_width / 2

    n_cols = 4
    n_rows = int(np.ceil(len(unique_projects) / n_cols))

    all_nn_counts  = {}
    all_rf_counts  = {}
    all_nn_correct = {}
    all_rf_correct = {}

    plt.rcParams.update({
        "font.size":        18,
        "axes.titlesize":   18,
        "axes.labelsize":   18,
        "xtick.labelsize":  18,
        "ytick.labelsize":  18,
        "legend.fontsize":  18,
    })

    for project in unique_projects:
        idx = np.where((projects == project).values)[0]

        project_nn_pred = nn_pred.iloc[idx].reset_index(drop=True)
        project_nn_true = nn_true.iloc[idx].reset_index(drop=True)
        project_rf_pred = rf_pred.iloc[idx].reset_index(drop=True)

        nn_overlap_scores = []
        nn_correct_counts = {}
        for i in range(len(project_nn_pred)):
            top3_nn   = set(project_nn_pred.iloc[i].nlargest(3).index.tolist())
            top3_true = set(project_nn_true.iloc[i].nlargest(3).index.tolist())
            correct   = top3_nn & top3_true
            nn_overlap_scores.append(len(correct) / 3)
            for sig in correct:
                nn_correct_counts[sig] = nn_correct_counts.get(sig, 0) + 1

        rf_overlap_scores = []
        rf_correct_counts = {}
        for i in range(len(project_rf_pred)):
            top3_rf   = set(project_rf_pred.iloc[i].nlargest(3).index.tolist())
            top3_true = set(project_nn_true.iloc[i].nlargest(3).index.tolist())
            correct   = top3_rf & top3_true
            rf_overlap_scores.append(len(correct) / 3)
            for sig in correct:
                rf_correct_counts[sig] = rf_correct_counts.get(sig, 0) + 1

        nn_ov = np.array(nn_overlap_scores)
        rf_ov = np.array(rf_overlap_scores)

        all_nn_counts[project]  = [np.isclose(nn_ov, cat).sum() for cat in categories]
        all_rf_counts[project]  = [np.isclose(rf_ov, cat).sum() for cat in categories]
        all_nn_correct[project] = nn_correct_counts
        all_rf_correct[project] = rf_correct_counts

    # ── Figure 1: top-3 overlap ──────────────────────────────────────────────
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes1 = axes1.flatten()

    for proj_idx, project in enumerate(unique_projects):
        ax = axes1[proj_idx]

        nn_counts = all_nn_counts[project]
        rf_counts = all_rf_counts[project]
        x = np.array(categories)

        b1 = ax.bar(x + offset_nn, nn_counts, width=bar_width,
                    color=color_nn, label="Hist2Sig", edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + offset_rf, rf_counts, width=bar_width,
                    color=color_rf, label="RF tumor types", edgecolor="white", linewidth=0.8)

        for bar, count in zip(b1, nn_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(int(count)),
                        ha="center", fontsize=8, color=color_nn, fontweight="bold")

        for bar, count in zip(b2, rf_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(int(count)),
                        ha="center", fontsize=8, color=color_rf, fontweight="bold")

        ax.set_facecolor("#F7F7F7")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{project}", fontweight="bold", fontsize=13)
        ax.set_xlabel("Top-3 Overlap Score", fontsize=15)
        ax.set_ylabel("Number of Samples", fontsize=15)
        ax.set_xticks(categories)
        ax.set_xticklabels(category_labels)
        ax.legend(loc="upper left", fontsize="x-small")

    for i in range(len(unique_projects), len(axes1)):
        axes1[i].axis("off")

    fig1.tight_layout()
    if save_dir is not None:
        fig1.savefig(f"{save_dir}/plot1_topk_overlap.pdf", bbox_inches="tight")
    plt.show()

    # ── Figure 2: signature diversity ────────────────────────────────────────
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
    axes2 = axes2.flatten()

    for proj_idx, project in enumerate(unique_projects):
        ax = axes2[proj_idx]

        nn_correct_counts = all_nn_correct[project]
        rf_correct_counts = all_rf_correct[project]
        all_sigs = set(nn_correct_counts) | set(rf_correct_counts)

        if all_sigs:
            sorted_sigs = sorted(all_sigs,
                                 key=lambda s: nn_correct_counts.get(s, 0),
                                 reverse=True)
            nn_vals = [nn_correct_counts.get(s, 0) for s in sorted_sigs]
            colors  = [V_BLUE if s in rf_correct_counts else V_VIOLET for s in sorted_sigs]

            ax.barh(sorted_sigs, nn_vals,
                    color=colors, edgecolor="white", linewidth=0.8)

            for i, sig in enumerate(sorted_sigs):
                nn_c = nn_correct_counts.get(sig, 0)
                rf_c = rf_correct_counts.get(sig, 0)
                ax.text(max(nn_c, rf_c) + 0.2, i,
                        f"H:{nn_c} | RF:{rf_c}",
                        va="center", fontsize=14, fontweight="bold")

        ax.set_facecolor("#F7F7F7")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{project}", fontweight="bold", fontsize=14)
        ax.invert_yaxis()

        is_last_row = (proj_idx // n_cols) == (n_rows - 1)
        if is_last_row:
            ax.set_xlabel("Correct Identification Count", fontsize=14)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=True)

    legend_elements = [
        Patch(facecolor=V_VIOLET, label="Hist2Sig only"),
        Patch(facecolor=V_BLUE,   label="Also identified by RF tumor types"),
    ]
    fig2.legend(handles=legend_elements,
                loc="lower center",
                ncol=2,
                fontsize=15,
                framealpha=0.9,
                bbox_to_anchor=(0.5, -0.02))

    for i in range(len(unique_projects), len(axes2)):
        axes2[i].axis("off")

    fig2.tight_layout()
    if save_dir is not None:
        fig2.savefig(f"{save_dir}/plot2_signature_diversity.pdf", bbox_inches="tight")
    plt.show()

    return {
        "nn_pred":         nn_pred,
        "nn_true":         nn_true,
        "rf_pred":         rf_pred,
        "projects":        projects,
        "unique_projects": unique_projects,
        "all_nn_counts":   all_nn_counts,
        "all_rf_counts":   all_rf_counts,
        "all_nn_correct":  all_nn_correct,
        "all_rf_correct":  all_rf_correct,
    }


# ════════════════════════════════════════════════════════════════════════════
# 8. Mean top-3 overlap with Wilcoxon (TCGA)
# ════════════════════════════════════════════════════════════════════════════

def plot_mean_overlap_tcga(nn_true, nn_pred, rf_pred, projects, unique_projects,
                           save_dir=None):
    """Mean top-3 overlap per tumour type, Hist2Sig vs RF (cell 15)."""
    V_VIOLET, V_BLUE = model_colors()
    color_nn = V_VIOLET
    color_rf = V_BLUE

    nn_overlap_per_project = {}
    rf_overlap_per_project = {}

    for project in unique_projects:
        idx = np.where((projects == project).values)[0]

        project_nn_pred = nn_pred.iloc[idx].reset_index(drop=True)
        project_nn_true = nn_true.iloc[idx].reset_index(drop=True)
        project_rf_pred = rf_pred.iloc[idx].reset_index(drop=True)

        nn_scores = []
        rf_scores = []
        for i in range(len(project_nn_pred)):
            top3_true = set(project_nn_true.iloc[i].nlargest(3).index.tolist())
            top3_nn   = set(project_nn_pred.iloc[i].nlargest(3).index.tolist())
            top3_rf   = set(project_rf_pred.iloc[i].nlargest(3).index.tolist())
            nn_scores.append(len(top3_nn & top3_true) / 3)
            rf_scores.append(len(top3_rf & top3_true) / 3)

        nn_overlap_per_project[project] = np.array(nn_scores)
        rf_overlap_per_project[project] = np.array(rf_scores)

    mean_nn = {p: nn_overlap_per_project[p].mean() for p in unique_projects}
    mean_rf = {p: rf_overlap_per_project[p].mean() for p in unique_projects}

    sem_nn  = {p: nn_overlap_per_project[p].std() / np.sqrt(len(nn_overlap_per_project[p]))
               for p in unique_projects}
    sem_rf  = {p: rf_overlap_per_project[p].std() / np.sqrt(len(rf_overlap_per_project[p]))
               for p in unique_projects}

    diff = {p: mean_nn[p] - mean_rf[p] for p in unique_projects}
    sorted_projects_diff = sorted(unique_projects, key=lambda p: diff[p], reverse=True)

    nn_vals   = [mean_nn[p]  for p in sorted_projects_diff]
    rf_vals   = [mean_rf[p]  for p in sorted_projects_diff]
    nn_sems   = [sem_nn[p]   for p in sorted_projects_diff]
    rf_sems   = [sem_rf[p]   for p in sorted_projects_diff]
    diff_vals = [diff[p]     for p in sorted_projects_diff]

    raw_pvals = []
    for p in sorted_projects_diff:
        nn_s = nn_overlap_per_project[p]
        rf_s = rf_overlap_per_project[p]
        d    = nn_s - rf_s
        if np.all(d == 0):
            raw_pvals.append(1.0)
        else:
            _, pval = wilcoxon(nn_s, rf_s)
            raw_pvals.append(pval)

    _, pvals_fdr, _, _ = multipletests(raw_pvals, method='fdr_bh')

    marks = [sig_mark(p) for p in pvals_fdr]

    n         = len(sorted_projects_diff)
    bar_width = 0.35
    gap       = 0.10

    fig, (ax, ax_diff) = plt.subplots(2, 1,
                                      figsize=(max(24, n * 0.85), 14),
                                      height_ratios=[3, 1], sharex=True)
    x = np.arange(n)

    ax.bar(x - bar_width/2 - gap/2, nn_vals, width=bar_width,
           color=color_nn, label="Hist2Sig", edgecolor="white", linewidth=0.6,
           yerr=nn_sems, capsize=3,
           error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2))

    ax.bar(x + bar_width/2 + gap/2, rf_vals, width=bar_width,
           color=color_rf, label="RF tumor types", edgecolor="white", linewidth=0.6,
           yerr=rf_sems, capsize=3,
           error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2))

    for i, (vn, vr, sn, sr) in enumerate(zip(nn_vals, rf_vals, nn_sems, rf_sems)):
        ax.text(i - bar_width/2 - gap/2, vn + sn + 0.03, f"{vn:.2f}",
                ha="center", va="bottom", fontsize=14, color=color_nn,
                fontweight="bold", rotation=90)
        ax.text(i + bar_width/2 + gap/2, vr + sr + 0.03, f"{vr:.2f}",
                ha="center", va="bottom", fontsize=14, color=color_rf,
                fontweight="bold", rotation=90)

    for i, (vn, vr, sn, sr, mark) in enumerate(
            zip(nn_vals, rf_vals, nn_sems, rf_sems, marks)):
        y_annot = max(vn + sn, vr + sr) + 0.20
        x_left  = i - bar_width/2 - gap/2
        x_right = i + bar_width/2 + gap/2
        ax.plot([x_left, x_left, x_right, x_right],
                [y_annot - 0.01, y_annot, y_annot, y_annot - 0.01],
                lw=0.8, color='#333333')
        ax.text(i, y_annot + 0.01, mark,
                ha='center', va='bottom', fontsize=12, color='#333333')

    ax.set_ylabel("Mean Top-3 Overlap", fontsize=18)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="upper right", fontsize=18, framealpha=0.95)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlim(-0.8, n - 0.2)

    colors_diff = [color_nn if d >= 0 else "#d73027" for d in diff_vals]
    ax_diff.bar(x, diff_vals, width=bar_width * 2 + gap,
                color=colors_diff, edgecolor="white", linewidth=0.6, alpha=0.85)
    ax_diff.axhline(0, color="black", linewidth=0.8)

    for i, d in enumerate(diff_vals):
        va     = "bottom" if d >= 0 else "top"
        offset = 0.02 if d >= 0 else -0.02
        ax_diff.text(i, d + offset, f"{d:+.2f}",
                     ha="center", va=va, fontsize=15, fontweight="bold",
                     color=colors_diff[i])

    y_margin = max(abs(d) for d in diff_vals) * 0.5
    ax_diff.set_ylim(min(diff_vals) - y_margin, max(diff_vals) + y_margin)
    ax_diff.set_ylabel("Δ Overlap\n(Hist2Sig − RF)", fontsize=18)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(sorted_projects_diff, fontsize=18,
                            rotation=45, ha="right")
    ax_diff.grid(False)
    ax_diff.set_facecolor("white")
    for spine in ax_diff.spines.values():
        spine.set_visible(False)
    ax_diff.tick_params(axis="x", length=0)
    ax_diff.tick_params(axis="y", labelsize=18)

    plt.subplots_adjust(hspace=0.09)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/plot_option3_mean_overlap.pdf", bbox_inches="tight")
    plt.show()

    return pd.DataFrame({
        "Project ID": sorted_projects_diff,
        "mean_nn":    nn_vals,
        "mean_rf":    rf_vals,
        "diff":       diff_vals,
        "pval_fdr":   pvals_fdr,
        "mark":       marks,
    })


# ════════════════════════════════════════════════════════════════════════════
# 9. CPTAC data preparation
# ════════════════════════════════════════════════════════════════════════════

def build_cptac(dir_files, sig_dir='./Signatures_datasets'):
    """Load CPTAC exposures and align them with the refit predictions (cell 17).

    Returns a dict with ``case_map``, ``cptac_pred``, ``exposures``,
    ``CPTAC_tumor_type``, ``exp_trim``, ``sig_cols``, ``pred_median`` and the
    four aligned frames.
    """
    case_map = (pd.read_csv(f'{sig_dir}/cptac_case_id_check.csv')
                  .set_index('Case ID')['matched_folders'])

    cptac_pred = pd.read_csv(f'./{dir_files}/cptac_exposures_refit.csv').drop(columns='Case ID')
    cptac_pred['Case ID'] = cptac_pred.pop('Image ID').str.split('-').str[:2].str.join('-')
    cptac_pred['Project ID'] = cptac_pred['Case ID'].map(case_map)

    exposures = (pd.read_csv(f'{sig_dir}/CPTAC_exposures_v1.csv')
                   .set_index('Case ID')
                   .rename(columns={'cases.primary_site': 'Primary Type'}))
    CPTAC_tumor_type = exposures['Primary Type'].tolist()

    exp_trim = exposures.iloc[:, 3:]
    sig_cols = exp_trim.columns[8:-1]

    pred_median = cptac_pred.groupby('Case ID').median(numeric_only=True)[sig_cols]

    def build_eval(true_exp):
        """Align an exposures subset with the predicted medians and tag both with Project ID."""
        idx  = true_exp.index.intersection(pred_median.index)
        true = true_exp.loc[idx].rename(columns={'Project ID': 'noname'})
        true.insert(0, 'Project ID', case_map.reindex(true.index))
        pred = pred_median.loc[idx].copy()
        pred['Project ID'] = true['Project ID']
        return true, pred

    true_activities_cptac, pred_cptac_median_eval = build_eval(exp_trim)

    gat_mask = exposures['File Name'].str.contains("GATK4_MuTect2_Pair", case=False, na=False)
    true_activities_cptac_gat, pred_cptac_gat_eval = build_eval(exp_trim[gat_mask])

    return {
        "case_map":                  case_map,
        "cptac_pred":                cptac_pred,
        "exposures":                 exposures,
        "CPTAC_tumor_type":          CPTAC_tumor_type,
        "exp_trim":                  exp_trim,
        "sig_cols":                  sig_cols,
        "pred_median":               pred_median,
        "true_activities_cptac":     true_activities_cptac,
        "pred_cptac_median_eval":    pred_cptac_median_eval,
        "true_activities_cptac_gat": true_activities_cptac_gat,
        "pred_cptac_gat_eval":       pred_cptac_gat_eval,
    }


def rf_predict_cptac(true_activities_cptac_gat, encoder, final_model, y_cols):
    """Apply the tumour-type RF to the CPTAC cohort (cell 18).

    Returns ``(proj_cptac, X_cptac, rf_pred_cptac)``.
    """
    proj_cptac = true_activities_cptac_gat["Project ID"].replace(CPTAC_TO_TCGA)

    X_cptac = encoder.transform(proj_cptac.values.reshape(-1, 1))
    assert X_cptac.sum(axis=1).min() == 1, "some rows are all zero"

    rf_pred_cptac = pd.DataFrame(
        final_model.predict(X_cptac),
        columns=[c.replace("true_", "") for c in y_cols],
        index=true_activities_cptac_gat.index,
    )

    return proj_cptac, X_cptac, rf_pred_cptac


# ════════════════════════════════════════════════════════════════════════════
# 10. CPTAC signature landscape (bubble plot)
# ════════════════════════════════════════════════════════════════════════════

def plot_bubble_cptac(true_activities_cptac_gat, save_path=None):
    """Signature landscape across CPTAC tumour types (cell 19)."""
    df = true_activities_cptac_gat.iloc[:, 9:-1].copy()
    df["Project ID"] = true_activities_cptac_gat["Project ID"]
    sbs_cols = [col for col in df.columns if col.startswith("SBS")]

    df_norm = df.copy()
    row_sums = df[sbs_cols].sum(axis=1)
    df_norm[sbs_cols] = df[sbs_cols].div(row_sums, axis=0).fillna(0)

    results = []
    for tumour, group in df.groupby("Project ID"):
        n = len(group)
        group_norm = df_norm.loc[group.index]
        for sbs in sbs_cols:
            results.append({
                "Tumour":     tumour,
                "SBS":        sbs,
                "MeanProp":   group_norm[sbs].mean(),
                "PctPresent": (group[sbs] > 0).sum() / n * 100,
            })

    bubble_df   = pd.DataFrame(results)
    pivot_size  = bubble_df.pivot(index="SBS", columns="Tumour", values="MeanProp")
    pivot_color = bubble_df.pivot(index="SBS", columns="Tumour", values="PctPresent")

    tumour_counts = df.groupby("Project ID").size().sort_values(ascending=False)
    pivot_size    = pivot_size[tumour_counts.index]
    pivot_color   = pivot_color[tumour_counts.index]

    sorted_sbs  = sorted(pivot_size.index, key=sbs_sort_key)
    pivot_size  = pivot_size.loc[sorted_sbs]
    pivot_color = pivot_color.loc[sorted_sbs]

    keep        = (pivot_size > 0).any(axis=1)
    pivot_size  = pivot_size.loc[keep]
    pivot_color = pivot_color.loc[keep]

    signatures = list(pivot_size.index)
    tumours    = list(pivot_size.columns)
    n_t = len(tumours)
    n_s = len(signatures)

    # figure geometry (inches converted to figure fractions)
    col_width   = 0.9
    plot_area_w = max(2.5, n_t * col_width)
    left_inch   = 2.0
    right_inch  = 3.0
    fig_w = plot_area_w + left_inch + right_inch
    fig_h = max(10, n_s * 0.42) + 2.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    left   = left_inch / fig_w
    right  = 1.0 - (right_inch / fig_w)
    bottom = 0.05

    header_inch = 1.8
    header_frac = header_inch / fig_h
    gap_frac    = 0.005
    top_plot    = 1.0 - header_frac - gap_frac
    plot_h      = top_plot - bottom

    ax = fig.add_axes([left, bottom, right - left, plot_h])

    cmap       = cm.viridis
    norm_color = mcolors.Normalize(vmin=0, vmax=100)
    max_size   = 500

    for i, sig in enumerate(signatures):
        for j, tumour in enumerate(tumours):
            prop = pivot_size.loc[sig, tumour]
            pct  = pivot_color.loc[sig, tumour]
            if prop > 0:
                ax.scatter(j, i, s=prop * max_size,
                           color=cmap(norm_color(pct)),
                           edgecolors="none", zorder=3)

    ax.set_xticks(range(n_t))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False)

    ax.set_yticks(range(n_s))
    ax.set_yticklabels([make_ylabel(s, ETIOLOGY_CPTAC) for s in signatures], fontsize=12)
    ax.set_ylabel("Mutational signature", fontsize=15)

    ax.set_xlim(-0.5, n_t - 0.5)
    ax.set_ylim(-0.5, n_s - 0.5)
    ax.grid(True, color="lightgray", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # header: sample counts + tumour names
    ax_top = fig.add_axes([left, top_plot + gap_frac, right - left, header_frac])
    ax_top.set_xlim(-0.5, n_t - 0.5)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    circle_y = 0.10
    name_y   = 0.28

    for j, tumour in enumerate(tumours):
        n = tumour_counts[tumour]
        ax_top.text(j, circle_y, str(n),
                    ha="center", va="center",
                    fontsize=8, color="navy",
                    bbox=dict(boxstyle="circle,pad=0.35",
                              fc="lightblue", ec="steelblue", lw=1))
        ax_top.text(j, name_y, tumour,
                    ha="left", va="bottom",
                    fontsize=9, rotation=40, color="black")

    cbar_left   = right + 0.02
    cbar_bottom = bottom + 0.25
    cbar_h      = plot_h * 0.45
    ax_cbar     = fig.add_axes([cbar_left, cbar_bottom, 0.02, cbar_h])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("% of samples with signature", fontsize=11,
                   rotation=90, labelpad=15)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.tick_params(labelsize=10)

    legend_props   = [0.25, 0.50, 0.75, 1.0]
    legend_handles = [
        plt.scatter([], [], s=p * max_size, color="gray",
                    edgecolors="none", label=f"{p:.2f}")
        for p in legend_props
    ]
    legend_bottom = cbar_bottom + cbar_h + 0.04
    ax_leg = fig.add_axes([cbar_left - 0.01, legend_bottom, 0.10, 0.20])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=legend_handles,
        title="Normalized\nexposure",
        loc="upper left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        scatterpoints=1,
        borderpad=0.8,
        labelspacing=0.8,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 11. Cosine similarity CPTAC: Hist2Sig vs RF
# ════════════════════════════════════════════════════════════════════════════

def plot_cosine_cptac_with_rf(true_activities_cptac_gat, pred_cptac_gat_eval,
                              rf_pred_cptac, save_dir=None):
    """Cosine similarity per tumour type, Hist2Sig vs RF, on CPTAC (cell 20)."""
    idx = (true_activities_cptac_gat.index
           .intersection(pred_cptac_gat_eval.index)
           .intersection(rf_pred_cptac.index))

    df_true = true_activities_cptac_gat.loc[idx].iloc[:, 9:-1]
    df_nn   = pred_cptac_gat_eval.loc[idx].iloc[:, :-1]
    df_rf   = rf_pred_cptac.loc[idx]

    sigs = list(df_true.columns)
    assert set(df_nn.columns) == set(sigs), set(df_nn.columns) ^ set(sigs)
    assert set(df_rf.columns) == set(sigs), set(df_rf.columns) ^ set(sigs)
    df_nn = df_nn[sigs]
    df_rf = df_rf[sigs]

    project_ids = pred_cptac_gat_eval.loc[idx, "Project ID"].replace(TCGA_TO_SHORT)

    def make_cos_df(df_pred, model_label):
        a, b = df_true.values, df_pred.values
        cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        return pd.DataFrame({
            "Project ID":        project_ids.values,
            "cosine_similarity": cos,
            "Model":             model_label,
        })

    df_cos = pd.concat([
        make_cos_df(df_nn, "Hist2Sig"),
        make_cos_df(df_rf, "RF tumor types"),
    ], ignore_index=True)

    order = (
        df_cos[df_cos["Model"] == "Hist2Sig"]
        .groupby("Project ID")["cosine_similarity"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    counts = project_ids.value_counts()
    labels = [f"{p}\n(n={counts[p]})" for p in order]

    stats = {}
    for proj in order:
        sel = df_cos["Project ID"] == proj
        nn_vals = df_cos.loc[sel & (df_cos["Model"] == "Hist2Sig"), "cosine_similarity"].values
        rf_vals = df_cos.loc[sel & (df_cos["Model"] == "RF tumor types"), "cosine_similarity"].values
        if np.all(nn_vals - rf_vals == 0):
            stats[proj] = (np.nan, "ns")
        else:
            _, p = wilcoxon(nn_vals, rf_vals)
            stats[proj] = (p, significance_mark(p))

    viridis_r = _viridis_r()
    palette = {
        "Hist2Sig":       matplotlib.colors.to_hex(viridis_r(0.92)),
        "RF tumor types": matplotlib.colors.to_hex(viridis_r(0.65)),
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.boxplot(
        data=df_cos,
        x="Project ID", y="cosine_similarity", hue="Model",
        order=order, hue_order=["Hist2Sig", "RF tumor types"],
        palette=palette, showfliers=False,
        linewidth=1.1, width=0.65, ax=ax,
    )
    sns.stripplot(
        data=df_cos,
        x="Project ID", y="cosine_similarity", hue="Model",
        order=order, hue_order=["Hist2Sig", "RF tumor types"],
        palette=palette, dodge=True,
        size=2.5, alpha=0.35, jitter=0.15,
        edgecolor="none", legend=False, ax=ax,
    )

    y_max     = df_cos["cosine_similarity"].max()
    bar_width = 0.65 / 2

    for i, proj in enumerate(order):
        _, mark = stats[proj]
        y_annot = y_max + 0.03
        x_left  = i - bar_width / 2
        x_right = i + bar_width / 2
        ax.plot([x_left, x_left, x_right, x_right],
                [y_annot - 0.01, y_annot, y_annot, y_annot - 0.01],
                lw=0.8, color="#333333")
        ax.text(i, y_annot + 0.005, mark,
                ha="center", va="bottom", fontsize=11, color="#333333")

    ax.set_ylim(0, y_max + 0.12)
    ax.set_xlabel("")
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=10)

    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labs[:2], title="Model", frameon=False,
              fontsize=10, title_fontsize=10, loc="lower left")

    sns.despine(ax=ax)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/cosine_similarity_cptac_con_rf.pdf", bbox_inches="tight")
    plt.show()

    print("\nWilcoxon signed-rank test (Hist2Sig vs RF, paired per patient):")
    for proj in order:
        p, mark = stats[proj]
        print(f"  {proj:6s}  p={p:.4f}  {mark}")

    return df_cos, stats


# ════════════════════════════════════════════════════════════════════════════
# 12. TCGA vs CPTAC cohort comparison
# ════════════════════════════════════════════════════════════════════════════

def prepare_cohort(df, cohort_name):
    df = df[df['Project ID'].isin(SELECTED_PROJECTS)].copy()
    df['Cohort'] = cohort_name
    return df


def _tmb_stats(tcga, cptac, sig_cols):
    stats_rows = []
    for proj in SELECTED_PROJECTS:
        a = tcga.loc[tcga['Project ID'] == proj, 'TMB'].values
        b = cptac.loc[cptac['Project ID'] == proj, 'TMB'].values
        if len(a) > 0 and len(b) > 0:
            u, p = mannwhitneyu(a, b, alternative='two-sided')
            stats_rows.append({
                'Project ID': proj,
                'n_TCGA': len(a),
                'n_CPTAC': len(b),
                'median_TCGA': float(np.median(a)),
                'median_CPTAC': float(np.median(b)),
                'U': float(u),
                'p_value': float(p),
            })
    return pd.DataFrame(stats_rows)


def plot_tmb_pred(tcga, cptac, sig_cols, out_path):
    """TMB comparison, large-font variant used by :func:`run_cohort_comparison`
    (cell 21). Saves to ``out_path`` and closes the figure."""
    tcga = tcga.copy()
    cptac = cptac.copy()
    tcga['TMB'] = tcga[sig_cols].sum(axis=1)
    cptac['TMB'] = cptac[sig_cols].sum(axis=1)

    combined = pd.concat([
        tcga[['Project ID', 'Cohort', 'TMB']],
        cptac[['Project ID', 'Cohort', 'TMB']],
    ], ignore_index=True)
    combined['Tumor type'] = combined['Project ID'].map(PROJECT_LABELS)

    stats_df = _tmb_stats(tcga, cptac, sig_cols)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(9, 5.2))
    order = [PROJECT_LABELS[p] for p in SELECTED_PROJECTS]

    sns.boxplot(
        data=combined, x='Tumor type', y='TMB', hue='Cohort',
        order=order, hue_order=['TCGA', 'CPTAC'],
        palette=COHORT_PALETTE, ax=ax,
        showfliers=False, linewidth=1.2, width=0.7,
    )
    sns.stripplot(
        data=combined, x='Tumor type', y='TMB', hue='Cohort',
        order=order, hue_order=['TCGA', 'CPTAC'],
        palette=COHORT_PALETTE, ax=ax,
        dodge=True, size=2.2, alpha=0.35, jitter=0.18,
        edgecolor='none', legend=False,
    )

    ax.set_yscale('log')
    ax.set_ylabel('(sum of signature exposures, log scale)',
                  fontsize=16)
    ax.set_xlabel('')
    ax.tick_params(axis='both', labelsize=16)

    y_top = combined['TMB'].max()
    for i, row in stats_df.iterrows():
        p = row['p_value']
        if p < 0.001:
            mark = '***'
        elif p < 0.01:
            mark = '**'
        elif p < 0.05:
            mark = '*'
        else:
            mark = 'ns'
        ax.text(i, y_top * 0.28, mark, ha='center', va='bottom',
                fontsize=11, color='#333333')

    ax.set_ylim(top=y_top * 3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title='Cohort', frameon=False,
              loc='upper right', fontsize=13, title_fontsize=15)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()

    return stats_df


def plot_composition_pred(tcga, cptac, sig_cols, min_contribution=0.02):
    """Mean relative composition, large-font variant (cell 21).

    The figure is left open (no ``savefig`` / ``close``) so Jupyter displays it,
    matching the notebook.
    """
    def mean_relative(df):
        sums = df[sig_cols].sum(axis=1)
        rel = df[sig_cols].div(sums.replace(0, np.nan), axis=0).fillna(0)
        rel['Project ID'] = df['Project ID'].values
        return rel.groupby('Project ID')[sig_cols].mean()

    tcga_comp = mean_relative(tcga).reindex(SELECTED_PROJECTS)
    cptac_comp = mean_relative(cptac).reindex(SELECTED_PROJECTS)

    max_per_sig = pd.concat([tcga_comp, cptac_comp]).max(axis=0)
    keep_sigs = max_per_sig[max_per_sig > min_contribution].index.tolist()
    keep_sigs = sorted(keep_sigs, key=_sig_sort_key)

    cmap = plt.get_cmap('tab20')
    sig_colors = {sig: cmap(i % 20) for i, sig in enumerate(keep_sigs)}
    colors_list = [sig_colors[s] for s in keep_sigs]

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    short_labels = [PROJECT_LABELS[p] for p in SELECTED_PROJECTS]

    for ax, comp_df, title in [
        (axes[0], tcga_comp, 'TCGA'),
        (axes[1], cptac_comp, 'CPTAC'),
    ]:
        comp_df = comp_df[keep_sigs]
        comp_df.index = short_labels
        comp_df.plot(
            kind='bar', stacked=True, ax=ax,
            color=colors_list, width=0.78,
            edgecolor='white', linewidth=0.6, legend=False,
        )
        ax.set_title(title, fontsize=16, weight='bold', pad=8)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0, labelsize=16)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Mean relative signature exposure', fontsize=16)

    handles = [plt.Rectangle((0, 0), 1, 1, color=sig_colors[s]) for s in keep_sigs]
    fig.legend(
        handles, keep_sigs,
        title='Signature', loc='center left',
        bbox_to_anchor=(0.99, 0.5), frameon=False,
        fontsize=14, title_fontsize=14, ncol=1,
    )

    plt.tight_layout()

    cos_rows = []
    for proj in SELECTED_PROJECTS:
        v1 = tcga_comp.loc[proj].values
        v2 = cptac_comp.loc[proj].values
        cs = 1.0 - cosine(v1, v2)
        cos_rows.append({'Project ID': proj, 'cosine_similarity': float(cs)})
    cos_df = pd.DataFrame(cos_rows)

    return tcga_comp, cptac_comp, cos_df


def run_cohort_comparison(true_activities, cptac_gat_anal, save_dir):
    """TMB + composition comparison between cohorts (driver of cell 21)."""
    sig_cols = get_signature_columns(true_activities)
    assert sig_cols == get_signature_columns(cptac_gat_anal), \
        "Signature columns differ between cohorts"

    tcga = prepare_cohort(true_activities, 'TCGA')
    cptac = prepare_cohort(cptac_gat_anal, 'CPTAC')

    print("=== 1. TMB comparison ===")
    tmb_stats = plot_tmb_pred(tcga, cptac, sig_cols,
                              out_path=f'{save_dir}/tmb_comparison_pred.pdf')
    print(tmb_stats.to_string(index=False))

    print("\n=== 2. Signature composition ===")
    tcga_comp, cptac_comp, cos_df = plot_composition_pred(tcga, cptac, sig_cols)
    print("\nCosine similarity between mean profiles (TCGA vs CPTAC):")
    print(cos_df.to_string(index=False))

    return {
        'tmb_stats': tmb_stats,
        'tcga_comp': tcga_comp,
        'cptac_comp': cptac_comp,
        'cosine': cos_df,
    }


def plot_tmb_real_vs_pred(true_tcga, true_cptac, pred_tcga, pred_cptac,
                          sig_cols, out_path):
    """Two-panel boxplot: left = observed, right = predicted (cell 22).
    Shared y axis (log) so the two panels can be compared directly."""

    def _tmb_long(df, cohort):
        df = df[df['Project ID'].isin(SELECTED_PROJECTS)].copy()
        df['TMB'] = df[sig_cols].sum(axis=1)
        df['Cohort'] = cohort
        df['Tumor type'] = df['Project ID'].map(PROJECT_LABELS)
        return df[['Tumor type', 'Cohort', 'TMB']]

    obs = pd.concat([
        _tmb_long(true_tcga, 'TCGA'),
        _tmb_long(true_cptac, 'CPTAC'),
    ], ignore_index=True)
    prd = pd.concat([
        _tmb_long(pred_tcga, 'TCGA'),
        _tmb_long(pred_cptac, 'CPTAC'),
    ], ignore_index=True)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    order = [PROJECT_LABELS[p] for p in SELECTED_PROJECTS]

    for ax, data, title in [
        (axes[0], obs, 'Real'),
        (axes[1], prd, 'Predicted'),
    ]:
        sns.boxplot(
            data=data, x='Tumor type', y='TMB', hue='Cohort',
            order=order, hue_order=['TCGA', 'CPTAC'],
            palette=COHORT_PALETTE, ax=ax,
            showfliers=False, linewidth=1.2, width=0.7,
        )
        sns.stripplot(
            data=data, x='Tumor type', y='TMB', hue='Cohort',
            order=order, hue_order=['TCGA', 'CPTAC'],
            palette=COHORT_PALETTE, ax=ax,
            dodge=True, size=2.2, alpha=0.35, jitter=0.18,
            edgecolor='none', legend=False,
        )
        ax.set_yscale('log')

        ax.set_title(title, fontsize=20, weight='bold', pad=8)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=18)
        sns.despine(ax=ax)

        # remove duplicated legend on each panel; we add a single one outside
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[0].set_ylabel('Signature mutation load\n(sum of exposures, log scale)',
                       fontsize=18)
    axes[1].set_ylabel('')

    # shared y limits = union of both panels, with headroom
    y_top = max(obs['TMB'].max(), prd['TMB'].max())
    y_bot = min(obs.loc[obs['TMB'] > 0, 'TMB'].min(),
                prd.loc[prd['TMB'] > 0, 'TMB'].min())
    axes[0].set_ylim(bottom=y_bot * 0.7, top=y_top/4)

    # single shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COHORT_PALETTE['TCGA']),
        plt.Rectangle((0, 0), 1, 1, color=COHORT_PALETTE['CPTAC']),
    ]
    fig.legend(
        handles, ['TCGA', 'CPTAC'],
        title='Cohort', loc='center left',
        bbox_to_anchor=(0.995, 0.5), frameon=False,
        fontsize=18, title_fontsize=18,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.show()

    # summary stats per cohort/tumor type for both observed and predicted
    summary = (
        pd.concat([obs.assign(Kind='Observed'), prd.assign(Kind='Predicted')])
        .groupby(['Kind', 'Cohort', 'Tumor type'])['TMB']
        .agg(['median', 'mean', 'count'])
        .reset_index()
    )
    return summary


def run_tmb_real_vs_pred(true_tcga, true_cptac, pred_tcga, pred_cptac, save_dir):
    """Stand-alone driver for the observed-vs-predicted TMB figure (cell 22)."""
    sig_cols = get_signature_columns(true_tcga)
    for df, name in [(true_cptac, 'true_cptac'),
                     (pred_tcga, 'pred_tcga'),
                     (pred_cptac, 'pred_cptac')]:
        assert get_signature_columns(df) == sig_cols, \
            f"Signature columns differ in {name}"

    print("=== TMB observed vs predicted ===")
    summary = plot_tmb_real_vs_pred(true_tcga, true_cptac, pred_tcga, pred_cptac,
                                    sig_cols,
                                    out_path=f'{save_dir}/tmb_real_vs_pred.pdf')
    return summary


# ════════════════════════════════════════════════════════════════════════════
# 13. Pearson bubble plot (CPTAC)
# ════════════════════════════════════════════════════════════════════════════

def plot_pearson_bubble_cptac(true_activities_cptac_gat, pred_cptac_gat_eval,
                              proj_cptac,
                              r_threshold=0.2, min_n=10, min_n_tumor=10,
                              min_prevalence=3, save_dir=None):
    """Per-tumour, per-signature Pearson r on CPTAC, positive r only (cell 24)."""
    BIN_EDGES   = [0.2, 0.3, 0.5, 0.7, 1.01]
    BIN_LABELS  = ["0.2 – 0.3", "0.3 – 0.5", "0.5 – 0.7", "> 0.7"]
    BUBBLE_SIZE = 320

    viridis_r  = _viridis_r()
    BIN_COLORS = [matplotlib.colors.to_hex(viridis_r(v)) for v in [0.10, 0.35, 0.65, 0.92]]

    # data: the three frames must come from the same cohort
    idx = true_activities_cptac_gat.index.intersection(pred_cptac_gat_eval.index)

    df_true = true_activities_cptac_gat.loc[idx].iloc[:, 9:-1]
    df_pred = pred_cptac_gat_eval.loc[idx].iloc[:, :-1]

    project_ids = proj_cptac.loc[idx].replace(TCGA_TO_CPTAC_SHORT)

    all_sigs = [c for c in df_true.columns if c.startswith("SBS")]
    assert set(all_sigs) <= set(df_pred.columns), set(all_sigs) - set(df_pred.columns)
    assert len(project_ids) == len(df_true) == len(df_pred)

    tumors         = project_ids.unique()
    tumors_to_plot = [t for t in tumors if (project_ids == t).sum() >= min_n_tumor]

    records  = []
    n_skipped_prevalence = 0

    for tumor in tumors_to_plot:
        pos      = np.where((project_ids == tumor).values)[0]
        true_raw = df_true.iloc[pos][all_sigs].reset_index(drop=True)
        pred_raw = df_pred.iloc[pos][all_sigs].reset_index(drop=True)
        t_norm, p_norm = normalize_by_sample(true_raw, pred_raw, all_sigs)

        for sig in all_sigs:
            t_v, p_v = t_norm[sig], p_norm[sig]
            n = len(t_v)
            prevalence = int((t_v > 0).sum())

            if n < min_n or t_v.std() == 0 or p_v.std() == 0:
                continue
            if prevalence < min_prevalence:
                n_skipped_prevalence += 1
                continue

            r, pval = pearsonr(t_v, p_v)
            records.append(dict(tumor=tumor, sig=sig, r=r, pval=pval,
                                n=n, prevalence=prevalence))

    df = pd.DataFrame(records)
    _, df["pval_fdr"], _, _ = multipletests(df["pval"], method="fdr_bh")

    print(f"Tested {len(df)} tumour x signature pairs; "
          f"{n_skipped_prevalence} skipped for prevalence < {min_prevalence}")

    neg = df[df["r"] <= -r_threshold]
    print(f"Excluded {len(neg)} negative correlations with r <= -{r_threshold} "
          f"(FDR<0.05: {(neg['pval_fdr'] < 0.05).sum()})")

    # positive correlations only
    df_sig = df[df["r"] >= r_threshold].copy()

    sig_order = sorted(df_sig["sig"].unique().tolist(), key=sig_sort_key)
    tumor_order = (df_sig.groupby("tumor")["sig"].count()
                   .sort_values(ascending=False).index.tolist())

    r_pivot    = df_sig.pivot_table(index="sig", columns="tumor", values="r",
                                    aggfunc="first").reindex(index=sig_order, columns=tumor_order)
    pval_pivot = df_sig.pivot_table(index="sig", columns="tumor", values="pval_fdr",
                                    aggfunc="first").reindex(index=sig_order, columns=tumor_order)

    n_sigs   = len(sig_order)
    n_tumors = len(tumor_order)

    fig_w = max(8, n_tumors * 0.65 + 2.5)
    fig_h = max(5, n_sigs * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    def r_to_bin(r_val):
        for k in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[k] <= r_val < BIN_EDGES[k + 1]:
                return k
        return len(BIN_COLORS) - 1

    for i, sig in enumerate(sig_order):
        for j, tumor in enumerate(tumor_order):
            r_val = r_pivot.loc[sig, tumor]
            p_val = pval_pivot.loc[sig, tumor]
            if np.isnan(r_val):
                continue
            ax.scatter(j, i, s=BUBBLE_SIZE, color=BIN_COLORS[r_to_bin(r_val)],
                       edgecolors="white", linewidths=0.5, zorder=3)

            if not np.isnan(p_val):
                stars = ("***" if p_val < 0.001 else
                         "**"  if p_val < 0.01  else
                         "*"   if p_val < 0.05  else "")
                if stars:
                    ax.text(j, i, stars, ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold", zorder=4)

    ax.set_xticks(range(n_tumors))
    ax.set_xticklabels(tumor_order, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(n_sigs))
    ax.set_yticklabels(sig_order, fontsize=12)
    ax.set_xlim(-0.7, n_tumors - 0.3)
    ax.set_ylim(-0.7, n_sigs - 0.3)
    ax.set_facecolor("#F7F7F7")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="white", linewidth=0.8, zorder=0)

    legend_handles = [
        plt.scatter([], [], s=BUBBLE_SIZE, color=BIN_COLORS[k],
                    edgecolors="white", linewidths=0.5, label=BIN_LABELS[k])
        for k in range(len(BIN_LABELS))
    ]
    ax.legend(handles=legend_handles, title="Pearson r", loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=11, title_fontsize=12,
              frameon=False, scatterpoints=1)

    ax.set_title(
        f"Pearson between real and predicted exposures for each tumor type and mutational signature\n"
        f"(r ≥ {r_threshold}, signature present in ≥ {min_prevalence} samples, "
        f"*, **, *** = FDR < 0.05 / 0.01 / 0.001)",
        fontsize=13, pad=14
    )
    ax.set_xlabel("Tumor type", fontsize=13)
    ax.set_ylabel("Signature", fontsize=13)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/bubble_pearson_tumor_sig.png", bbox_inches="tight", dpi=300)
    plt.show()

    return df


# ════════════════════════════════════════════════════════════════════════════
# 14. Top-3 overlap + signature diversity (CPTAC)
# ════════════════════════════════════════════════════════════════════════════

def top3_figures_cptac(true_activities_cptac_gat, pred_cptac_gat_eval,
                       final_model, X_cptac, y_cols,
                       save_dir=None, save_fig1=True, save_fig2=False):
    """Top-3 overlap histograms and signature diversity on CPTAC (cell 25).

    Returns a dict with the objects later cells need.
    """
    V_VIOLET, V_BLUE = model_colors()
    color_nn = V_VIOLET
    color_rf = V_BLUE

    # data: one index shared by true / Hist2Sig / RF
    idx = true_activities_cptac_gat.index.intersection(pred_cptac_gat_eval.index)

    nn_true = true_activities_cptac_gat.loc[idx].iloc[:, 9:-1]
    sbs_cols = [c for c in nn_true.columns if c.startswith("SBS")]
    nn_true = nn_true[sbs_cols]

    nn_pred = pred_cptac_gat_eval.loc[idx].iloc[:, :-1]

    rf_sig_names = [c.replace("true_", "") for c in y_cols]
    rf_pred = pd.DataFrame(final_model.predict(X_cptac),
                           columns=rf_sig_names,
                           index=true_activities_cptac_gat.index).loc[idx]

    assert set(sbs_cols) <= set(nn_pred.columns), set(sbs_cols) - set(nn_pred.columns)
    assert set(sbs_cols) <= set(rf_pred.columns), set(sbs_cols) - set(rf_pred.columns)
    nn_pred = nn_pred[sbs_cols]
    rf_pred = rf_pred[sbs_cols]

    projects = true_activities_cptac_gat.loc[idx, "Project ID"]
    assert len(nn_true) == len(nn_pred) == len(rf_pred) == len(projects)

    unique_projects = projects.unique()

    categories      = [0, 1/3, 2/3, 1]
    category_labels = ["0", "1/3", "2/3", "1"]

    bar_width = 0.12
    offset_nn = -bar_width / 2
    offset_rf =  bar_width / 2

    n_cols = 4
    n_rows = int(np.ceil(len(unique_projects) / n_cols))

    all_nn_counts  = {}
    all_rf_counts  = {}
    all_nn_correct = {}
    all_rf_correct = {}

    for project in unique_projects:
        pos = np.where((projects == project).values)[0]

        project_nn_true = nn_true.iloc[pos].reset_index(drop=True)
        project_nn_pred = nn_pred.iloc[pos].reset_index(drop=True)
        project_rf_pred = rf_pred.iloc[pos].reset_index(drop=True)

        nn_overlap_scores, nn_correct_counts = [], {}
        rf_overlap_scores, rf_correct_counts = [], {}

        for i in range(len(project_nn_true)):
            top3_true = set(project_nn_true.iloc[i].nlargest(3).index)
            top3_nn   = set(project_nn_pred.iloc[i].nlargest(3).index)
            top3_rf   = set(project_rf_pred.iloc[i].nlargest(3).index)

            correct_nn = top3_nn & top3_true
            nn_overlap_scores.append(len(correct_nn) / 3)
            for sig in correct_nn:
                nn_correct_counts[sig] = nn_correct_counts.get(sig, 0) + 1

            correct_rf = top3_rf & top3_true
            rf_overlap_scores.append(len(correct_rf) / 3)
            for sig in correct_rf:
                rf_correct_counts[sig] = rf_correct_counts.get(sig, 0) + 1

        nn_ov = np.array(nn_overlap_scores)
        rf_ov = np.array(rf_overlap_scores)

        all_nn_counts[project]  = [np.isclose(nn_ov, cat).sum() for cat in categories]
        all_rf_counts[project]  = [np.isclose(rf_ov, cat).sum() for cat in categories]
        all_nn_correct[project] = nn_correct_counts
        all_rf_correct[project] = rf_correct_counts

    # ── Figure 1: top-3 overlap ──────────────────────────────────────────────
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
                               constrained_layout=True)
    axes1 = axes1.flatten()

    for proj_idx, project in enumerate(unique_projects):
        ax = axes1[proj_idx]

        nn_counts = all_nn_counts[project]
        rf_counts = all_rf_counts[project]
        x = np.array(categories)

        b1 = ax.bar(x + offset_nn, nn_counts, width=bar_width,
                    color=color_nn, label="Hist2Sig", edgecolor="white", linewidth=0.8)
        b2 = ax.bar(x + offset_rf, rf_counts, width=bar_width,
                    color=color_rf, label="RF tumor types", edgecolor="white", linewidth=0.8)

        for bar, count in zip(b1, nn_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(int(count)),
                        ha="center", fontsize=8, color=color_nn, fontweight="bold")

        for bar, count in zip(b2, rf_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(int(count)),
                        ha="center", fontsize=8, color=color_rf, fontweight="bold")

        ax.set_facecolor("#F7F7F7")
        for spine in ax.spines.values():
            spine.set_visible(False)

        label = project.split("-")[1] if "-" in project else project
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.set_xticks(categories)
        ax.set_xticklabels(category_labels)

        is_last_row = (proj_idx // n_cols) == (n_rows - 1)
        ax.set_xlabel("Top-3 Overlap Score" if is_last_row else "", fontsize=10)

    for i in range(len(unique_projects), len(axes1)):
        axes1[i].axis("off")

    fig1.legend(handles=[Patch(facecolor=V_VIOLET, label="Hist2Sig"),
                         Patch(facecolor=V_BLUE,   label="RF tumor types")],
                loc="lower center", ncol=2, fontsize=11,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig1.suptitle("Top-3 Overlap: Hist2Sig vs RF tumor types — CPTAC",
                  fontsize=16, fontweight="bold")
    if save_dir is not None and save_fig1:
        fig1.savefig(f"{save_dir}/plot1_topk_overlap_CPTAC.pdf", bbox_inches="tight")
    plt.show()

    # ── Figure 2: signature diversity ────────────────────────────────────────
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
                               constrained_layout=True)
    axes2 = axes2.flatten()

    for proj_idx, project in enumerate(unique_projects):
        ax = axes2[proj_idx]

        nn_correct_counts = all_nn_correct[project]
        rf_correct_counts = all_rf_correct[project]
        all_sigs = set(nn_correct_counts) | set(rf_correct_counts)

        if all_sigs:
            sorted_sigs = sorted(all_sigs,
                                 key=lambda s: nn_correct_counts.get(s, 0),
                                 reverse=True)
            nn_vals = [nn_correct_counts.get(s, 0) for s in sorted_sigs]
            colors  = [V_BLUE if s in rf_correct_counts else V_VIOLET for s in sorted_sigs]

            ax.barh(sorted_sigs, nn_vals,
                    color=colors, edgecolor="white", linewidth=0.8)

            for i, sig in enumerate(sorted_sigs):
                nn_c = nn_correct_counts.get(sig, 0)
                rf_c = rf_correct_counts.get(sig, 0)
                ax.text(max(nn_c, rf_c) + 0.2, i,
                        f"H:{nn_c} | RF:{rf_c}",
                        va="center", fontsize=8, fontweight="bold")

        ax.set_facecolor("#F7F7F7")
        for spine in ax.spines.values():
            spine.set_visible(False)

        label = project.split("-")[1] if "-" in project else project
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.invert_yaxis()

        is_last_row = (proj_idx // n_cols) == (n_rows - 1)
        ax.set_xlabel("Correct Identification Count" if is_last_row else "", fontsize=10)

    for i in range(len(unique_projects), len(axes2)):
        axes2[i].axis("off")

    fig2.legend(handles=[Patch(facecolor=V_VIOLET, label="Hist2Sig only"),
                         Patch(facecolor=V_BLUE,   label="Also identified by RF tumor types")],
                loc="lower center", ncol=2, fontsize=11,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig2.suptitle("Signature Diversity: Hist2Sig vs RF tumor types — CPTAC",
                  fontsize=16, fontweight="bold")
    if save_dir is not None and save_fig2:
        fig2.savefig(f"{save_dir}/plot2_signature_diversity_CPTAC.pdf", bbox_inches="tight")
    plt.show()

    return {
        "nn_true":         nn_true,
        "nn_pred":         nn_pred,
        "rf_pred":         rf_pred,
        "projects":        projects,
        "unique_projects": unique_projects,
        "all_nn_counts":   all_nn_counts,
        "all_rf_counts":   all_rf_counts,
        "all_nn_correct":  all_nn_correct,
        "all_rf_correct":  all_rf_correct,
    }


# ════════════════════════════════════════════════════════════════════════════
# 15. Mean top-3 overlap with Wilcoxon (CPTAC)
# ════════════════════════════════════════════════════════════════════════════

def plot_mean_overlap_cptac(nn_true, nn_pred, rf_pred, projects, unique_projects,
                            save_dir=None):
    """Mean top-3 overlap per tumour type on CPTAC, with SEM and Wilcoxon (cell 26)."""
    V_VIOLET, V_BLUE = model_colors()
    color_nn = V_VIOLET
    color_rf = V_BLUE

    # sample-level overlap (for Wilcoxon test + mean/sem)
    nn_overlap_per_project = {}
    rf_overlap_per_project = {}

    for project in unique_projects:

        idx = np.where((projects == project).values)[0]

        project_nn_pred = nn_pred.iloc[idx].reset_index(drop=True)
        project_nn_true = nn_true.iloc[idx].reset_index(drop=True)
        project_rf_pred = rf_pred.iloc[idx].reset_index(drop=True)

        nn_scores = []
        rf_scores = []

        for i in range(len(project_nn_pred)):

            top3_true = set(project_nn_true.iloc[i].nlargest(3).index.tolist())
            top3_nn   = set(project_nn_pred.iloc[i].nlargest(3).index.tolist())
            top3_rf   = set(project_rf_pred.iloc[i].nlargest(3).index.tolist())

            nn_scores.append(len(top3_nn & top3_true) / 3)
            rf_scores.append(len(top3_rf & top3_true) / 3)

        nn_overlap_per_project[project] = np.array(nn_scores)
        rf_overlap_per_project[project] = np.array(rf_scores)

    # mean + SEM per project
    mean_nn, sem_nn = {}, {}
    mean_rf, sem_rf = {}, {}

    for p in unique_projects:
        nn_m, nn_s = mean_sem(nn_overlap_per_project[p])
        rf_m, rf_s = mean_sem(rf_overlap_per_project[p])

        mean_nn[p], sem_nn[p] = nn_m, nn_s
        mean_rf[p], sem_rf[p] = rf_m, rf_s

    # sorting by difference
    diff = {p: mean_nn[p] - mean_rf[p] for p in unique_projects}
    sorted_projects_diff = sorted(unique_projects, key=lambda p: diff[p], reverse=True)

    short_labels = [p.split('-')[1] if '-' in p else p for p in sorted_projects_diff]

    nn_vals = [mean_nn[p] for p in sorted_projects_diff]
    rf_vals = [mean_rf[p] for p in sorted_projects_diff]
    nn_errs = [sem_nn[p] for p in sorted_projects_diff]
    rf_errs = [sem_rf[p] for p in sorted_projects_diff]
    diff_vals = [diff[p] for p in sorted_projects_diff]

    # Wilcoxon + FDR correction
    raw_pvals = []

    for p in sorted_projects_diff:

        nn_s = nn_overlap_per_project[p]
        rf_s = rf_overlap_per_project[p]

        d = nn_s - rf_s

        if np.all(d == 0):
            raw_pvals.append(1.0)
        else:
            _, pval = wilcoxon(nn_s, rf_s)
            raw_pvals.append(pval)

    _, pvals_fdr, _, _ = multipletests(raw_pvals, method='fdr_bh')

    marks = [sig_mark(p) for p in pvals_fdr]

    n = len(sorted_projects_diff)

    fig, (ax, ax_diff) = plt.subplots(
        2, 1,
        figsize=(max(8, n * 1.6), 10),
        height_ratios=[3, 1],
        sharex=True,
    )

    x = np.arange(n)
    bar_width = 0.30
    gap = 0.08

    # top panel (mean + SEM)
    ax.bar(
        x - bar_width/2 - gap/2,
        nn_vals,
        width=bar_width,
        color=color_nn,
        label="Hist2Sig",
        edgecolor="white",
        linewidth=0.6,
        yerr=nn_errs,
        capsize=4,
        error_kw=dict(lw=1, capthick=1)
    )

    ax.bar(
        x + bar_width/2 + gap/2,
        rf_vals,
        width=bar_width,
        color=color_rf,
        label="RF tumor types",
        edgecolor="white",
        linewidth=0.6,
        yerr=rf_errs,
        capsize=4,
        error_kw=dict(lw=1, capthick=1)
    )

    # value labels
    for i, (vn, vr) in enumerate(zip(nn_vals, rf_vals)):

        ax.text(
            i - bar_width/2 - gap/2,
            vn + nn_errs[i] + 0.01,
            f"{vn:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=color_nn,
            fontweight="bold",
            rotation=90
        )

        ax.text(
            i + bar_width/2 + gap/2,
            vr + rf_errs[i] + 0.01,
            f"{vr:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=color_rf,
            fontweight="bold",
            rotation=90
        )

    # significance annotations
    for i, (vn, vr, mark) in enumerate(zip(nn_vals, rf_vals, marks)):

        y_annot = max(vn + nn_errs[i], vr + rf_errs[i]) + 0.15

        x_left  = i - bar_width/2 - gap/2
        x_right = i + bar_width/2 + gap/2

        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_annot - 0.01, y_annot, y_annot, y_annot - 0.01],
            lw=0.8,
            color='#333333'
        )

        ax.text(
            i,
            y_annot + 0.01,
            mark,
            ha='center',
            va='bottom',
            fontsize=12,
            color='#333333'
        )

    ax.set_ylabel("Mean Top-3 Overlap", fontsize=14)
    ax.set_ylim(0, 1.35)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.95)

    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.6, n - 0.4)

    # bottom panel (difference)
    colors_diff = [color_nn if d >= 0 else "#d73027" for d in diff_vals]

    ax_diff.bar(
        x,
        diff_vals,
        width=bar_width * 2 + gap,
        color=colors_diff,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85
    )

    ax_diff.axhline(0, color="black", linewidth=0.8)

    for i, d in enumerate(diff_vals):
        va = "bottom" if d >= 0 else "top"
        offset = 0.02 if d >= 0 else -0.02

        ax_diff.text(
            i,
            d + offset,
            f"{d:+.2f}",
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            color=colors_diff[i]
        )

    y_margin = max(abs(d) for d in diff_vals) * 0.5
    ax_diff.set_ylim(
        min(diff_vals) - y_margin,
        max(diff_vals) + y_margin
    )

    ax_diff.set_ylabel("Δ Overlap\n(Hist2Sig − RF)", fontsize=14)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(short_labels, fontsize=14)

    ax_diff.grid(False)
    ax_diff.set_facecolor("white")
    for spine in ax_diff.spines.values():
        spine.set_visible(False)

    ax_diff.tick_params(axis="x", length=0)
    ax_diff.tick_params(axis="y", labelsize=13)

    plt.subplots_adjust(hspace=0.08)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/plot_mean_overlap_CPTAC.pdf", bbox_inches="tight")
    plt.show()

    return pd.DataFrame({
        "Project ID": sorted_projects_diff,
        "mean_nn":    nn_vals,
        "sem_nn":     nn_errs,
        "mean_rf":    rf_vals,
        "sem_rf":     rf_errs,
        "diff":       diff_vals,
        "pval_fdr":   pvals_fdr,
        "mark":       marks,
    })


# ════════════════════════════════════════════════════════════════════════════
# 16. Signature diversity for a subset of tumour types
# ════════════════════════════════════════════════════════════════════════════

def plot_signature_diversity_subset(all_nn_correct, all_rf_correct,
                                    selected=('TCGA-GBM', 'TCGA-PAAD'),
                                    short=None, save_path=None):
    """Signature diversity restricted to a few tumour types (cell 27)."""
    V_VIOLET, V_BLUE = model_colors()

    selected = list(selected)
    
    # accept either TCGA project ids or CPTAC short codes
    available = set(all_nn_correct)
    tcga_to_cptac = {v: k for k, v in CPTAC_TO_TCGA.items()}
    resolved = []
    for p in selected:
        if p in available:
            resolved.append(p)
        elif tcga_to_cptac.get(p) in available:
            resolved.append(tcga_to_cptac[p])
        elif CPTAC_TO_TCGA.get(p) in available:
            resolved.append(CPTAC_TO_TCGA[p])
        else:
            raise KeyError(f"{p!r} not found; available: {sorted(available)}")
    if short is not None:
        short = {r: short.get(r, short.get(p, r)) for r, p in zip(resolved, selected)}
    selected = resolved
    
    if short is None:
        short = {p: (p.split('-')[1] if '-' in p else p) for p in selected}

    fig, axes = plt.subplots(1, len(selected), figsize=(12, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, project in zip(axes, selected):
        nn_correct_counts = all_nn_correct[project]
        rf_correct_counts = all_rf_correct[project]
        all_sigs = set(nn_correct_counts) | set(rf_correct_counts)

        if all_sigs:
            sorted_sigs = sorted(all_sigs,
                                 key=lambda s: nn_correct_counts.get(s, 0),
                                 reverse=True)
            nn_vals = [nn_correct_counts.get(s, 0) for s in sorted_sigs]
            colors = [V_BLUE if s in rf_correct_counts else V_VIOLET
                      for s in sorted_sigs]

            ax.barh(sorted_sigs, nn_vals,
                    color=colors, edgecolor='white', linewidth=0.8)

            for i, sig in enumerate(sorted_sigs):
                nn_c = nn_correct_counts.get(sig, 0)
                rf_c = rf_correct_counts.get(sig, 0)
                ax.text(max(nn_c, rf_c) + 0.5, i,
                        f"H:{nn_c} | RF:{rf_c}",
                        va='center', fontsize=16, fontweight='bold')

        ax.set_facecolor('#F7F7F7')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(short[project], fontweight='bold', fontsize=16)
        ax.set_xlabel('\n Correct identification count', fontsize=16)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=16)

    legend_elements = [
        Patch(facecolor=V_VIOLET, label='Hist2Sig only'),
        Patch(facecolor=V_BLUE,   label='Also identified by RF'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=2, fontsize=16,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.15))

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
