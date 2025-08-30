import numpy as np


def route_length_np(route, D):
    r = np.asarray(route, dtype=int)
    return D[r, np.roll(r, -1)].sum()

def route_length(route, D):
    # fallback je ok, ali za naše dimenzije vektorizacija je čista dobit
    try:
        return float(route_length_np(route, D))
    except Exception:
        total = 0.0
        n = len(route)
        for i in range(n):
            a, b = route[i], route[(i + 1) % n]
            total += D[a, b]
        return total

def evaluate_route(route, D, fuel_per_km=1.0):
    dist = route_length(route, D)
    fuel = dist * fuel_per_km
    return dist, fuel

def fix_route(route, n=None):
    # robustnije od len(route): ako ima duplikata, len(route) može biti manji od n
    if n is None:
        n = int(np.max(route)) + 1
    seen, fixed = set(), []
    for g in route:
        gi = int(g) % n   # osigura da smo u [0, n-1]
        if gi not in seen:
            seen.add(gi)
            fixed.append(gi)
    for i in range(n):
        if i not in seen:
            fixed.append(i)
    return fixed

def route_from_genes(sol, n):
    """Za vektore gena (real/int) vrati validnu TSP permutaciju 0..n-1."""
    base = [int(round(g)) % n for g in sol]
    return fix_route(base, n)

def is_valid_tour(route, n):
    r = list(map(int, route))
    return (len(r) == n) and (set(r) == set(range(n)))

def tour_to_names(route, city_list):
    return [city_list[i][0] for i in route]

# Heuristike

def nn_route(D, start):
    n = D.shape[0]
    unused = set(range(n))
    route = [start]
    unused.remove(start)
    cur = start
    while unused:
        nxt = min(unused, key=lambda j: D[cur, j])
        route.append(nxt); unused.remove(nxt); cur = nxt
    return route

def two_opt_fast(route, D, tries_without_improve=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(route)
    best = route[:]
    best_len = route_length(best, D)
    noimp = 0
    while noimp < tries_without_improve:
        i = rng.integers(1, n-2)
        k = rng.integers(i+1, n-1)
        new_route = best[:]
        new_route[i:k+1] = reversed(best[i:k+1])
        new_len = route_length(new_route, D)
        if new_len + 1e-12 < best_len:
            best, best_len = new_route, new_len
            noimp = 0
        else:
            noimp += 1
    return best

def inversion_mutation(route, prob=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = route.copy()
    if rng.random() < prob and len(r) > 3:
        i = rng.integers(1, len(r)-2)
        j = rng.integers(i+1, len(r)-1)
        r[i:j+1] = r[i:j+1][::-1]
    return r

def canonical_start(route, start_idx=0):
    route = list(route)
    if start_idx not in route:   # zaštita
        return route
    i = route.index(start_idx)
    r = route[i:] + route[:i]
    return r if r[1] < r[-1] else [r[0]] + r[:0:-1]

# Rekombinacije (OX i ERX) 

def ordered_crossover(parents, offspring_size, ga):
    offspring = []
    num_genes   = ga.num_genes
    num_parents = parents.shape[0]
    for k in range(offspring_size[0]):
        p1 = parents[k % num_parents]
        p2 = parents[(k + 1) % num_parents]
        start = np.random.randint(0, num_genes - 1)
        end   = np.random.randint(start + 1, num_genes)
        child = [-1] * num_genes
        child[start:end] = p1[start:end].tolist()
        fill = [g for g in p2.tolist() if g not in child]
        idx = 0
        for i in range(num_genes):
            if child[i] == -1:
                child[i] = fill[idx]; idx += 1
        offspring.append(child)
    return np.array(offspring, dtype=int)

def _erx_build_maps(p1, p2):
    n = len(p1)
    adj = {i: set() for i in p1}
    common_adj = {i: set() for i in p1}
    def add_edges(path, dest):
        for i, node in enumerate(path):
            left  = path[(i - 1) % n]
            right = path[(i + 1) % n]
            dest[node].add(left); dest[node].add(right)
    add_edges(p1, adj); add_edges(p2, adj)
    nei1 = {i: set() for i in p1}; nei2 = {i: set() for i in p1}
    add_edges(p1, nei1); add_edges(p2, nei2)
    for i in p1:
        common_adj[i] = nei1[i].intersection(nei2[i])
    return adj, common_adj

def erx_crossover(parents, offspring_size, ga):
    num_parents = parents.shape[0]
    offspring = []
    rng = np.random.default_rng()
    for k in range(offspring_size[0]):
        p1 = parents[k % num_parents].astype(int).tolist()
        p2 = parents[(k + 1) % num_parents].astype(int).tolist()
        adj, common_adj = _erx_build_maps(p1, p2)
        remaining = set(p1)
        start = int(rng.choice([p1[0], p2[0]]))
        child, cur = [], start
        while remaining:
            child.append(cur); remaining.remove(cur)
            for s in adj.values():        s.discard(cur)
            for s in common_adj.values(): s.discard(cur)
            if not remaining: break
            cand = [v for v in common_adj[cur] if v in remaining]
            if not cand:
                cand = [v for v in adj[cur] if v in remaining]
            if cand:
                min_deg = min(len(adj[v]) for v in cand)
                bests = [v for v in cand if len(adj[v]) == min_deg]
                cur = int(rng.choice(bests))
            else:
                min_deg = min(len(adj[v]) for v in remaining)
                bests = [v for v in remaining if len(adj[v]) == min_deg]
                cur = int(rng.choice(bests))
        offspring.append(child)
    return np.array(offspring, dtype=int)


def erx_one_child(p1, p2, rng=None):
    """ERX: pravi JEDNO dete iz 2 roditelja-permutacije."""
    if rng is None: rng = np.random.default_rng()
    p1 = [int(x) for x in p1]; p2 = [int(x) for x in p2]
    adj, common_adj = _erx_build_maps(p1, p2)
    remaining = set(p1)
    start = rng.choice([p1[0], p2[0]])
    child = []
    cur = int(start)
    while remaining:
        child.append(cur)
        remaining.remove(cur)
        for s in adj.values():        s.discard(cur)
        for s in common_adj.values(): s.discard(cur)
        if not remaining: break
        cand = [v for v in common_adj[cur] if v in remaining]
        if not cand:
            cand = [v for v in adj[cur] if v in remaining]
        if cand:
            min_deg = min(len(adj[v]) for v in cand)
            bests = [v for v in cand if len(adj[v]) == min_deg]
            cur = int(rng.choice(bests))
        else:
            min_deg = min(len(adj[v]) for v in remaining)
            bests = [v for v in remaining if len(adj[v]) == min_deg]
            cur = int(rng.choice(bests))
    return child

def ox_child_with_best_segment(current, best, seg_len, rng=None):
    """OX: uzmi segment iz 'best', popuni ostatak po redosledu iz 'current'."""
    if rng is None: rng = np.random.default_rng()
    n = len(current)
    if seg_len < 2: seg_len = 2
    if seg_len > n-1: seg_len = n-1
    start = rng.integers(0, n - seg_len + 1)
    end = start + seg_len
    child = [-1]*n
    # segment iz best
    child[start:end] = best[start:end]
    used = set(best[start:end])
    fill = [g for g in current if g not in used]
    it = iter(fill)
    for i in range(n):
        if child[i] == -1:
            child[i] = next(it)
    return child