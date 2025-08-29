import argparse
import numpy as np

from tsp_data import cities, distance_matrix, fuel_per_km
from tsp_utils import nn_route, canonical_start, evaluate_route
from ga_runner import run_ga
from io_utils import write_pergen_csv, write_summary_csv
from viz import plot_cities, plot_route, plot_convergence

def build_initial_population(D, pop_size=80, nn_count=6, use_2opt_on_nn=False, seed=42):
    np.random.seed(seed)
    n = D.shape[0]
    starts = np.linspace(0, n-1, nn_count, dtype=int)
    nn_pop = [np.array(nn_route(D, s), dtype=int) for s in starts]
    if use_2opt_on_nn:
        from tsp_utils import two_opt_fast
        nn_pop = [np.array(two_opt_fast(r.tolist(), D, tries_without_improve=60), dtype=int) for r in nn_pop]
    rand_count = pop_size - len(nn_pop)
    rand_pop = [np.random.permutation(n) for _ in range(rand_count)]
    return np.array(nn_pop + rand_pop, dtype=int)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="GA", choices=["GA"], help="Koji algoritam da pokrenem")
    parser.add_argument("--gens", type=int, default=500)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="GA_ERX_inversion_2opt")
    parser.add_argument("--nn", type=int, default=6, help="broj NN tura u inicijalnoj populaciji")
    parser.add_argument("--twoopt_nn", action="store_true", help="primeni 2-opt na NN ture pri inicijalizaciji")
    args = parser.parse_args()

    D = distance_matrix
    n = D.shape[0]

    # 1) inicijalna populacija
    initial_population = build_initial_population(D, pop_size=args.pop, nn_count=args.nn, use_2opt_on_nn=args.twoopt_nn, seed=args.seed)

    # 2) prikaz gradova (opciono)
    plot_cities()

    # 3) pokretanje GA
    res = run_ga(
        initial_population=initial_population,
        D=D,
        fuel_per_km=fuel_per_km,
        num_generations=args.gens,
        sol_per_pop=args.pop,
        seed=args.seed,
        label=args.label
    )

    # 4) konvergencija
    plot_convergence(res["hist_best"], res["hist_mean"], res["hist_median"])

    # 5) finalna ruta
    best_route = canonical_start(res["best_route"], start_idx=0)
    dist, fuel = evaluate_route(best_route, D, fuel_per_km)
    print("\nNajbolja ruta (indeksi):", best_route)
    print("Ukupna distanca:        ", round(dist, 2), "km")
    print("Ukupna potrošnja goriva:", round(fuel, 2), "l")

    plot_route(best_route, title="Najbolja pronađena ruta")

    # 6) CSV
    exp_label = args.label
    seed_lbl  = args.seed
    run_id    = 1

    pergen_path  = f"results/{exp_label}_seed{seed_lbl}_run{run_id}_pergen.csv"
    summary_path = f"results/{exp_label}_seed{seed_lbl}_run{run_id}_summary.csv"

    write_pergen_csv(pergen_path, res["hist_best"], res["hist_mean"], res["hist_median"])
    summary = {
        **res["params"],
        "cities": len(cities),
        "distance_km": round(dist, 4),
        "fuel_l": round(fuel, 4),
        "runtime_sec": round(res["runtime_sec"], 4),
        "start_city": cities[0][0],
        "fuel_per_km": fuel_per_km,
        "route_indices": best_route,
        "route_city_names": [cities[i][0] for i in best_route],
    }
    write_summary_csv(summary_path, summary)
    print(f"\nCSV sa generacijama: {pergen_path}")
    print(f"CSV sa sažetkom:     {summary_path}")

if __name__ == "__main__":
    main()
