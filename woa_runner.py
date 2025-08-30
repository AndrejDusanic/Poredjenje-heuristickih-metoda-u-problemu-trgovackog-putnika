# woa_runner.py
import time
import numpy as np
from tsp_utils import (
    route_length, fix_route, two_opt_fast,
    inversion_mutation, erx_one_child, ox_child_with_best_segment
)

def run_woa(
    initial_population,
    D,
    fuel_per_km,
    num_iterations=500,
    seed=42,
    label="WOA_ERX_OX_2opt",
    b=1.0,                # spiral parametar
    apply_twoopt_every=10,# povremeni lokalni polishing
    twoopt_topk=3,
):
    rng = np.random.default_rng(seed)

    pop = np.array([fix_route(ind) for ind in initial_population], dtype=int)
    n, pop_size = len(pop[0]), len(pop)

    # inicijalno oceni
    dists = np.array([route_length(ind, D) for ind in pop], dtype=float)
    best_idx = int(np.argmin(dists))
    best_route = pop[best_idx].tolist()
    best_dist = float(dists[best_idx])

    hist_best, hist_mean, hist_median = [], [], []

    T0 = 0.05 * (np.mean(dists) + 1e-9)   # skala ~ kilometrima

    t0 = time.perf_counter()
    for t in range(num_iterations):
        # WOA parametri
        a = 2.0 - 2.0 * (t / max(1, num_iterations - 1))   # 2 -> 0
        new_pop = []
        new_dists = []

        for i in range(pop_size):
            Xi = pop[i].tolist()

            p = rng.random()
            r = rng.random()
            A = 2.0*a*r - a
            C = 2.0*r

            if p < 0.5:
                # Encircling/search grana
                if abs(A) < 1.0:
                    # eksploatacija: “ka najboljem” - ERX(Xi, best)
                    child = erx_one_child(Xi, best_route, rng)
                else:
                    # eksploracija: “ka random kitu”
                    j = int(rng.integers(0, pop_size))
                    while j == i:
                        j = int(rng.integers(0, pop_size))
                    child = erx_one_child(Xi, pop[j].tolist(), rng)
            else:
                # Spiral: segment iz best, popuna iz Xi (OX)
                # dužina segmenta - exp(b*|l|) * (n * 0.1)
                l = rng.uniform(-1, 1)
                seg_len = max(2, int(np.clip(np.exp(b*abs(l))*0.1*n, 2, n-1)))
                child = ox_child_with_best_segment(Xi, best_route, seg_len, rng)

            # blaga mutacija (može da opada sa vremenom)
            pm = 0.2*(1.0 - t/num_iterations) + 0.05  # 0.25 -> 0.05
            child = inversion_mutation(child, prob=pm, rng=rng)

            # validacija i evaluacija
            child = fix_route(child)
            cd = route_length(child, D)

            T = T0 * (1.0 - t / max(1, num_iterations-1))
            worse = cd >= dists[i]
            accept = (not worse) or (rng.random() < np.exp(-(cd - dists[i]) / max(1e-12, T)))

            if accept:
                new_pop.append(np.array(child, dtype=int))
                new_dists.append(cd)
            else:
                new_pop.append(pop[i])
                new_dists.append(dists[i])


        pop = np.array(new_pop, dtype=int)
        dists = np.array(new_dists, dtype=float)

        # povremeni 2-opt polishing na najboljima
        if (apply_twoopt_every is not None) and (t > 0) and (t % apply_twoopt_every == 0):
            idx_sorted = np.argsort(dists)[:twoopt_topk]
            for idx in idx_sorted:
                improved = two_opt_fast(pop[idx].tolist(), D, tries_without_improve=80)
                imp_d = route_length(improved, D)
                if imp_d + 1e-12 < dists[idx]:
                    pop[idx] = np.array(improved, dtype=int)
                    dists[idx] = imp_d

        # update globalnog najboljeg
        cur_best_idx = int(np.argmin(dists))
        cur_best_dist = float(dists[cur_best_idx])
        if cur_best_dist + 1e-12 < best_dist:
            best_dist = cur_best_dist
            best_route = pop[cur_best_idx].tolist()

        # istorija
        hist_best.append(float(np.min(dists)))
        hist_mean.append(float(np.mean(dists)))
        hist_median.append(float(np.median(dists)))

        # (opciono) print napredak
        # print(f"Iter {t+1:3d} | best={hist_best[-1]:.2f} | mean={hist_mean[-1]:.2f}")

    # finalni 2-opt polishing
    best_route = two_opt_fast(best_route, D, tries_without_improve=200)
    best_dist = route_length(best_route, D)

    t1 = time.perf_counter()

    return {
        "best_route": best_route,
        "hist_best": hist_best,
        "hist_mean": hist_mean,
        "hist_median": hist_median,
        "runtime_sec": t1 - t0,
        "params": {
            "algo": "WOA",
            "pop_size": pop_size,
            "num_iterations": num_iterations,
            "b_spiral": b,
            "apply_twoopt_every": apply_twoopt_every,
            "twoopt_topk": twoopt_topk,
            "seed": seed,
        }
    }
