import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 1. Biologically structured plasmid simulation
# ============================================================
def generate_biological_plasmids(
    n=100,
    n_families=8,
    seed=42,
):
    """
    Simulate plasmids with:
      - lengths
      - latent family structure
      - hub/mobile-element effect for a few plasmids

    Returns a dict with plasmid metadata.
    """
    rng = np.random.default_rng(seed)

    # Assign each plasmid to a latent family
    families = rng.integers(0, n_families, size=n)

    # Family-specific mean lengths
    family_mean_lengths = rng.uniform(4e4, 1.8e5, size=n_families)

    # Log-normal plasmid lengths, clipped to a reasonable range
    lengths = np.zeros(n, dtype=float)
    for i in range(n):
        mean_len = family_mean_lengths[families[i]]
        log_mu = np.log(mean_len) - 0.5 * 0.35**2
        lengths[i] = rng.lognormal(mean=log_mu, sigma=0.35)

    lengths = np.clip(lengths, 2e4, 3e5)

    # A few "hub" plasmids with promiscuous mobile elements
    hub_flags = np.zeros(n, dtype=bool)
    hub_indices = rng.choice(n, size=max(3, n // 20), replace=False)
    hub_flags[hub_indices] = True

    return {
        "families": families,
        "lengths": lengths,
        "hub_flags": hub_flags,
    }


def generate_structured_distances(plasmids, seed=42):
    """
    Generate:
      - d1: symmetric DCJ-Indel-like distance matrix
      - d2: directed containment distance matrix in [0,1]

    Biological assumptions:
      - same-family plasmids tend to have lower d1 and lower d2
      - very different lengths increase d2
      - hub plasmids can create misleading containment signals
      - d2(i,j) is "fraction of plasmid i not contained in plasmid j"
    """
    rng = np.random.default_rng(seed)

    families = plasmids["families"]
    lengths = plasmids["lengths"]
    hub_flags = plasmids["hub_flags"]
    n = len(lengths)

    d1 = np.zeros((n, n), dtype=float)
    d2 = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            same_family = families[i] == families[j]

            len_i = lengths[i]
            len_j = lengths[j]
            rel_len_diff = abs(len_i - len_j) / max(len_i, len_j)

            # --------------------------
            # DCJ-Indel distance d1
            # --------------------------
            # Same family => lower expected structural distance
            if same_family:
                base_dcj = rng.poisson(lam=2.0 + 5.0 * rel_len_diff)
            else:
                base_dcj = rng.poisson(lam=7.0 + 8.0 * rel_len_diff)

            # Hub/mobile plasmids can still be structurally diverse
            if hub_flags[i] or hub_flags[j]:
                base_dcj += rng.poisson(lam=1.5)

            d1[i, j] = base_dcj
            d1[j, i] = base_dcj

            # --------------------------
            # Directed containment d2
            # d2(i,j) = proportion of i not contained in j
            # --------------------------
            for a, b in [(i, j), (j, i)]:
                len_a = lengths[a]
                len_b = lengths[b]
                same_fam_ab = families[a] == families[b]

                # If a is smaller than b, containment is more plausible
                size_factor = 1.0 - min(len_a, len_b) / max(len_a, len_b)

                if len_a <= len_b:
                    if same_fam_ab:
                        # Better containment if same family and size-compatible
                        missing = rng.beta(1.5, 8.0) + 0.45 * size_factor
                    else:
                        # Worse containment across families
                        missing = rng.beta(3.5, 3.0) + 0.35 * size_factor
                else:
                    # If a is larger than b, harder for a to be contained in b
                    if same_fam_ab:
                        missing = rng.beta(3.5, 2.5) + 0.55 * size_factor
                    else:
                        missing = rng.beta(5.0, 1.8) + 0.45 * size_factor

                # Hub plasmids can share mobile elements, artificially improving apparent containment
                if hub_flags[a] or hub_flags[b]:
                    missing -= rng.uniform(0.05, 0.18)

                d2[a, b] = np.clip(missing, 0.0, 1.0)

    np.fill_diagonal(d1, 0.0)
    np.fill_diagonal(d2, 0.0)
    return d1, d2


# ============================================================
# 2. Normalization functions for DCJ-Indel
# ============================================================
def normalize_dcj_minmax(d1):
    mask = ~np.eye(d1.shape[0], dtype=bool)
    vals = d1[mask]
    dmin = vals.min()
    dmax = vals.max()

    if dmax == dmin:
        out = np.zeros_like(d1)
    else:
        out = (d1 - dmin) / (dmax - dmin)

    np.fill_diagonal(out, 0.0)
    return out


def normalize_dcj_exp(d1, tau=None):
    """
    F(d) = 1 - exp(-d / tau)
    """
    mask = (~np.eye(d1.shape[0], dtype=bool)) & (d1 > 0)
    nonzero = d1[mask]

    if tau is None:
        tau = np.median(nonzero) if nonzero.size > 0 else 1.0

    out = 1.0 - np.exp(-d1 / tau)
    np.fill_diagonal(out, 0.0)
    return out, tau


def normalize_dcj_rank(d1):
    """
    Empirical CDF / rank normalization onto [0,1].
    Useful as an alternative F(.).
    """
    n = d1.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = d1[mask]

    sorted_vals = np.sort(vals)
    ranks = np.searchsorted(sorted_vals, d1, side="right")
    out = ranks / len(sorted_vals)

    np.fill_diagonal(out, 0.0)
    return out


# ============================================================
# 3. Combined distance and graph sampling
# ============================================================
def combined_distance(F_d1, d2):
    d = np.maximum(F_d1, d2)
    np.fill_diagonal(d, 0.0)
    return d


def distance_to_probability(d, gamma=1.0):
    """
    Smaller distance => larger inclusion probability
    """
    p = (1.0 - d) ** gamma
    np.fill_diagonal(p, 0.0)
    return p


def sample_directed_graph(prob_matrix, seed=42):
    rng = np.random.default_rng(seed)
    n = prob_matrix.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    draws = rng.random((n, n))
    include = draws < prob_matrix
    np.fill_diagonal(include, False)

    rows, cols = np.where(include)
    for i, j in zip(rows, cols):
        G.add_edge(i, j, weight=float(prob_matrix[i, j]))

    return G


# ============================================================
# 4. Plotting helpers
# ============================================================
def plot_matrix(mat, title, cmap="viridis"):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="value")
    plt.title(title)
    plt.xlabel("target plasmid")
    plt.ylabel("source plasmid")
    plt.tight_layout()


def draw_graph(G, title, families=None, seed=42, node_size=90):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=seed, k=0.45 / np.sqrt(max(1, G.number_of_nodes())))

    if families is None:
        node_colors = None
    else:
        node_colors = families

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.9,
        cmap=plt.cm.tab10,
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=10,
        alpha=0.22,
        width=[0.4 + 1.8 * w for w in weights] if weights else 0.5,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


# ============================================================
# 5. Main demo
# ============================================================
def main():
    n = 100

    plasmids = generate_biological_plasmids(
        n=n,
        n_families=8,
        seed=123,
    )

    d1, d2 = generate_structured_distances(plasmids, seed=456)

    F_minmax = normalize_dcj_minmax(d1)
    F_exp, tau = normalize_dcj_exp(d1)
    F_rank = normalize_dcj_rank(d1)

    d_comb_minmax = combined_distance(F_minmax, d2)
    d_comb_exp = combined_distance(F_exp, d2)
    d_comb_rank = combined_distance(F_rank, d2)

    p_minmax = distance_to_probability(d_comb_minmax, gamma=1.2)
    p_exp = distance_to_probability(d_comb_exp, gamma=1.2)
    p_rank = distance_to_probability(d_comb_rank, gamma=1.2)

    G_minmax = sample_directed_graph(p_minmax, seed=11)
    G_exp = sample_directed_graph(p_exp, seed=22)
    G_rank = sample_directed_graph(p_rank, seed=33)

    print(f"n plasmids: {n}")
    print(f"median-based tau for exponential normalization: {tau:.3f}")
    print(f"minmax graph edges: {G_minmax.number_of_edges()}")
    print(f"exp graph edges:    {G_exp.number_of_edges()}")
    print(f"rank graph edges:   {G_rank.number_of_edges()}")

    plot_matrix(d1, "Raw DCJ-Indel matrix d1")
    plot_matrix(d2, "Directed containment distance matrix d2")
    plot_matrix(d_comb_exp, f"Combined distance max(1-exp(-d1/{tau:.2f}), d2)")
    plot_matrix(d_comb_rank, "Combined distance max(rank-normalized d1, d2)")

    draw_graph(
        G_exp,
        "Directed graph using exponential DCJ normalization",
        families=plasmids["families"],
        seed=1,
    )
    draw_graph(
        G_rank,
        "Directed graph using rank/CDF-like DCJ normalization",
        families=plasmids["families"],
        seed=2,
    )
    plt.savefig("figures/example_network_exp.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()