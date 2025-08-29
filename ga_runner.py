import time
import numpy as np
import pygad

from tsp_utils import (
    fix_route, route_length, two_opt_fast,
    erx_crossover
)

def _apply_two_opt_to_topk(ga, D, k=3):
    n = ga.num_genes
    dists, routes = [], []
    for sol in ga.population:
        r = fix_route([int(x) % n for x in sol], n=n)
        routes.append(r)
        dists.append(route_length(r, D))
    idx_sorted = np.argsort(dists)[:k]
    for idx in idx_sorted:
        improved = two_opt_fast(routes[idx], D, tries_without_improve=200)
        ga.population[idx] = np.array(improved, dtype=int)
    ga.cal_pop_fitness()

def _inject_immigrants(ga, n_cities, D, n_imm=2):
    dists = []
    for sol in ga.population:
        r = fix_route(sol, n=n_cities)
        dists.append(route_length(r, D))
    worst_idx = np.argsort(dists)[-n_imm:]
    for idx in worst_idx:
        ga.population[idx] = np.random.permutation(n_cities)
    ga.cal_pop_fitness()

def run_ga(initial_population,
           D,
           fuel_per_km,
           num_generations=500,
           sol_per_pop=80,
           num_parents_mating=20,
           mutation_type="inversion",
           mutation_percent_genes=8,
           parent_selection_type="tournament",
           K_tournament=3,
           crossover=erx_crossover,
           keep_parents=2,
           seed=42,
           label="GA_ERX"):
    #Vraća dict sa:
    #  hist_best, hist_mean, hist_median, best_route, best_distance, best_fuel, runtime_sec, params
    
    np.random.seed(seed)
    n = D.shape[0]

    hist_best, hist_mean, hist_median = [], [], []

    def fitness_func(ga, solution, sol_idx):
        r = fix_route([int(round(g)) % n for g in solution], n=n)
        dist = route_length(r, D)
        return 1.0 / (dist + 1e-6)

    STAG = 20
    def on_generation(ga_inst):
        # metrika posle modifikacija
        def pop_dists():
            return [route_length(fix_route([int(x)%n for x in sol], n=n), D)
                    for sol in ga_inst.population]
        # anti-stagnacija
        if len(hist_best) >= STAG and hist_best[-STAG] <= hist_best[-1] + 1e-9:
            _apply_two_opt_to_topk(ga_inst, D, k=3)
            _inject_immigrants(ga_inst, n, D, n_imm=2)
        dists = pop_dists()
        best, mean, med = float(np.min(dists)), float(np.mean(dists)), float(np.median(dists))
        hist_best.append(best); hist_mean.append(mean); hist_median.append(med)
        print(f"Gen {ga_inst.generations_completed:3d} | best={best:.2f} km | mean={mean:.2f}")

    ga = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        initial_population=initial_population,
        fitness_func=fitness_func,
        num_genes=n,
        gene_type=int,
        gene_space=list(range(n)),
        allow_duplicate_genes=False,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        crossover_type=crossover,
        on_generation=on_generation,
        stop_criteria=["saturate_30"],
        keep_parents=keep_parents,
        parent_selection_type=parent_selection_type,
        K_tournament=K_tournament,
    )

    t0 = time.perf_counter()
    ga.run()
    t1 = time.perf_counter()

    sol, fit, _ = ga.best_solution()
    best_route = fix_route(sol, n=n)
    # završni 2-opt (malo "poliranje")
    best_route = two_opt_fast(best_route, D, tries_without_improve=300)
    best_distance = route_length(best_route, D)
    best_fuel = best_distance * fuel_per_km

    params = {
        "algo": f"{label}",
        "sol_per_pop": sol_per_pop,
        "num_parents_mating": num_parents_mating,
        "mutation_type": mutation_type,
        "mutation_percent_genes": mutation_percent_genes,
        "selection": parent_selection_type,
        "K_tournament": K_tournament,
        "allow_duplicate_genes": False,
        "generations_completed": int(ga.generations_completed),
        "seed": seed,
    }

    return {
        "hist_best": hist_best,
        "hist_mean": hist_mean,
        "hist_median": hist_median,
        "best_route": best_route,
        "best_distance": best_distance,
        "best_fuel": best_fuel,
        "runtime_sec": t1 - t0,
        "params": params,
        "ga": ga,
    }
