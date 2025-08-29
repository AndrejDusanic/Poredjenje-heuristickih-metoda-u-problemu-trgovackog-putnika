import numpy as np
import matplotlib.pyplot as plt
import itertools
import pygad
import os, csv, time, json

from math import radians, sin, cos, sqrt, atan2

# Lista gradova sa njihovim geografskim koordinatama (latitude, longitude)
cities = [
    ("Seattle",        47.6062, -122.3321),
    ("San Francisco",  37.7749, -122.4194),
    ("Los Angeles",    34.0522, -118.2437),
    ("San Diego",      32.7157, -117.1611),
    ("Las Vegas",      36.1699, -115.1398),
    ("Salt Lake City", 40.7608, -111.8910),
    ("Denver",         39.7392, -104.9903),
    ("San Antonio",    29.4241,  -98.4936),
    ("Houston",        29.7604,  -95.3698),
    ("New Orleans",    29.9511,  -90.0715),
    ("Miami",          25.7617,  -80.1918),
    ("Atlanta",        33.7490,  -84.3880),
    ("Oklahoma City",  35.4676,  -97.5164),
    ("Dallas",         32.7767,  -96.7970),
    ("Chicago",        41.8781,  -87.6298),
    ("Milwaukee",      43.0389,  -87.9065),
    ("Minneapolis",    44.9778,  -93.2650),
    ("Detroit",        42.3314,  -83.0458),
    ("Cleveland",      41.4993,  -81.6944),
    ("Cincinnati",     39.1031,  -84.5120),
    ("Pittsburgh",     40.4406,  -79.9959),
    ("Baltimore",      39.2904,  -76.6122),
    ("Philadelphia",   39.9526,  -75.1652),
    ("New York",       40.7128,  -74.0060),
    ("Boston",         42.3601,  -71.0589),
    ("Indianapolis",   39.7684,  -86.1581),
    ("St. Louis",      38.6270,  -90.1994),
    ("Phoenix",        33.4484, -112.0740)
]

# Za crtanje koristimo (longitude, latitude)
coordinates = np.array([(lon, lat) for _, lat, lon in cities])
latlons     = np.array([(lat, lon) for _, lat, lon in cities])


# Funkcija za rastojanje (haversine formula)
def haversine(c1, c2):
    R = 6371.0  # Zemljin poluprečnik u km
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

history_best_dist  = []
history_mean_dist  = []
history_median_dist = []


# Gradi simetričnu matricu rastojanja dimenzija n×n,
# n je broj gradova.
def calculate_distance_matrix(latlons):
    n = len(latlons)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = haversine(latlons[i], latlons[j])
        D[i, j] = D[j, i] = d
    return D

distance_matrix = calculate_distance_matrix(latlons)


# Evaluacija rute (distanca + gorivo)
fuel_per_km = 1.2 #potrošnja goriva u litrama po kilometru


def route_length(route, D): #duzina rute
    total = 0.0
    for i in range(len(route)):
        a, b = route[i], route[(i + 1) % len(route)]
        total += D[a, b]
    return total

# Funkcija za evaluaciju rute
# Za permutaciju indeksa gradova (route): računa ukupnu pređenu distancu i ukupno potrošeno gorivo
def evaluate_route(route):
    total_dist = total_fuel = 0.0
    n = len(route)
    for i in range(n):
        a, b = route[i], route[(i + 1) % n] # vraćamo se na početak (početni grad)
        d = distance_matrix[a, b]
        total_dist += d
        total_fuel += d * fuel_per_km
    return total_dist, total_fuel


# Validna permutacija: uklanja duplikate i dodaje preostale indekse
def fix_route(route):
    n = len(cities)
    seen, fixed = set(), []
    for g in route:
        gi = int(g)
        if gi not in seen:
            seen.add(gi)
            fixed.append(gi)
    for i in range(n):
        if i not in seen:
            fixed.append(i)
    return fixed


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

def two_opt_fast(route, D, tries_without_improve=200):
    n = len(route)
    best = route[:]
    best_len = route_length(best, D)
    noimp = 0
    while noimp < tries_without_improve:
        i = np.random.randint(1, n-2)
        k = np.random.randint(i+1, n-1)
        new_route = best[:]
        new_route[i:k+1] = reversed(best[i:k+1])
        new_len = route_length(new_route, D)
        if new_len + 1e-12 < best_len:
            best, best_len = new_route, new_len
            noimp = 0
        else:
            noimp += 1
    return best

def apply_two_opt_to_topk(ga, k=3):
    n = len(cities)
    # 1) izračunaj distance cele populacije direktno
    dists = []
    routes = []
    for sol in ga.population:
        r = fix_route([int(x) % n for x in sol])
        routes.append(r)
        dists.append(route_length(r, distance_matrix))
    # 2) popravi top-k
    idx_sorted = np.argsort(dists)[:k]
    for idx in idx_sorted:
        improved = two_opt_fast(routes[idx], distance_matrix, tries_without_improve=200)
        ga.population[idx] = np.array(improved, dtype=int)
    # 3) VAŽNO: osveži fitnese nakon ručnih izmena populacije
    ga.cal_pop_fitness()
    
# fitness = 1 / (dist + eps)
def fitness_func(ga, solution, sol_idx):
    route = [int(round(g)) % len(cities) for g in solution]
    route = fix_route(route)
    dist = route_length(route, distance_matrix)
    return 1.0 / (dist + 1e-6) # izbegavanje deljenja sa nulom
        
#pracenje konvergencije + diverzitet        
history_best_dist = []  # NOVO: prati konvergenciju po generacijama
BASE_MUT_PCT = 8
STAG_WINDOW  = 20

def inject_immigrants(ga_inst, n=2):
    # ubaci n novih random permutacija umesto najgorih
    dists = []
    for sol in ga_inst.population:
        r = fix_route(sol)
        dists.append(route_length(r, distance_matrix))
    worst_idx = np.argsort(dists)[-n:]
    for idx in worst_idx:
        ga_inst.population[idx] = np.random.permutation(len(cities))
        
        
def on_generation(ga_inst):
    n = len(cities)

    #  zabeleži pre-modifikacije
    def pop_dists():
        return [route_length(fix_route([int(x)%n for x in sol]), distance_matrix)
                for sol in ga_inst.population]
    # pre_best = min(pop_dists())   ako želiš da ispišeš pre-posle
    
    #proveri stagnaciju
    STAG = 20
    if len(history_best_dist) > STAG and history_best_dist[-STAG] <= history_best_dist[-1] + 1e-9:
        apply_two_opt_to_topk(ga_inst, k=3)   # već zove cal_pop_fitness()
        inject_immigrants(ga_inst, n=2); ga_inst.cal_pop_fitness()

    # posle modifikacija – ponovo izračunaj
    dists = pop_dists()
    best  = float(np.min(dists))
    mean  = float(np.mean(dists))
    med   = float(np.median(dists))

    history_best_dist.append(best)
    history_mean_dist.append(mean)
    history_median_dist.append(med)

    print(f"Gen {ga_inst.generations_completed:3d} | best={best:.2f} km | mean={mean:.2f}")

def ordered_crossover(parents, offspring_size, ga):
    offspring = []
    num_genes   = ga.num_genes
    num_parents = parents.shape[0]

    for k in range(offspring_size[0]):
        p1 = parents[k % num_parents]
        p2 = parents[(k + 1) % num_parents]

        # segment granica
        start = np.random.randint(0, num_genes - 1)
        end   = np.random.randint(start + 1, num_genes)

        # generišem potomka, kopirajući segment iz p1 i popunjavajući ga sa genima iz p2
        child = [-1] * num_genes
        # 1) segment iz p1
        child[start:end] = p1[start:end].tolist()
        # 2) popuni iz p2
        fill = [g for g in p2.tolist() if g not in child]
        idx = 0
        for i in range(num_genes):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1

        offspring.append(child)

    return np.array(offspring)


def _erx_build_maps(p1, p2):
    """Pravi 2 mape: adj (unija suseda) i common_adj (presek suseda)."""
    n = len(p1)
    adj = {i: set() for i in p1}
    common_adj = {i: set() for i in p1}

    def add_edges(path, dest):
        for i, node in enumerate(path):
            left  = path[(i - 1) % n]
            right = path[(i + 1) % n]
            dest[node].add(left)
            dest[node].add(right)

    # unija suseda
    add_edges(p1, adj); add_edges(p2, adj)

    # presek suseda
    nei1 = {i: set() for i in p1}; nei2 = {i: set() for i in p1}
    add_edges(p1, nei1); add_edges(p2, nei2)
    for i in p1:
        common_adj[i] = nei1[i].intersection(nei2[i])

    return adj, common_adj

def erx_crossover(parents, offspring_size, ga):
    """
    PyGAD custom crossover: ERX (Edge Recombination).
    - parents: np.ndarray [num_parents, num_genes]
    - offspring_size: (num_offspring, num_genes)
    - return: np.ndarray [num_offspring, num_genes] dtype=int
    """
    num_parents = parents.shape[0]
    num_genes = ga.num_genes
    offspring = []

    rng = np.random.default_rng()  # za stabilnu nasumičnost

    for k in range(offspring_size[0]):
        p1 = parents[k % num_parents].astype(int).tolist()
        p2 = parents[(k + 1) % num_parents].astype(int).tolist()

        adj, common_adj = _erx_build_maps(p1, p2)

        remaining = set(p1)
        # start: ili slučajno iz {p1[0], p2[0]} ili čvor sa najmanjim stepenom (heuristika)
        start_candidates = [p1[0], p2[0]]
        # ili: start = min(remaining, key=lambda x: len(adj[x]))
        start = int(rng.choice(start_candidates))

        child = []
        cur = start

        while remaining:
            child.append(cur)
            remaining.remove(cur)

            # ukloni 'cur' iz susedstva svih (da se stepeni smanjuju kroz iteracije)
            for s in adj.values():        s.discard(cur)
            for s in common_adj.values(): s.discard(cur)

            if not remaining:
                break

            # 1) preferiraj zajedničke susede iz oba roditelja
            cand = [v for v in common_adj[cur] if v in remaining]
            if not cand:
                # 2) ili bilo kog suseda iz unije
                cand = [v for v in adj[cur] if v in remaining]
            if cand:
                # biraj one sa najmanjim preostalim stepenom
                min_deg = min(len(adj[v]) for v in cand)
                bests = [v for v in cand if len(adj[v]) == min_deg]
                cur = int(rng.choice(bests))
            else:
                # 3) nema suseda – uzmi iz preostalih sa najmanjim stepenom
                min_deg = min(len(adj[v]) for v in remaining)
                bests = [v for v in remaining if len(adj[v]) == min_deg]
                cur = int(rng.choice(bests))

        offspring.append(child)

    return np.array(offspring, dtype=int)

def plot_cities():
    # Prikazuje sve gradove na 2D grafu
    # (x=longitude, y=latitude).
    plt.figure(figsize=(10, 6))
    for name, lat, lon in cities:
        plt.scatter(lon, lat, color='red', zorder=2)
        plt.text(lon + 0.5, lat + 0.5, name, fontsize=8, zorder=3)
    plt.title("Lokacije gradova (geografske koordinate)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

def plot_route(route, title="Najbolja pronađena ruta"):
    plt.figure(figsize=(10, 6))
    for i in range(len(route)):
        a, b = route[i], route[(i + 1) % len(route)]
        lon1, lat1 = coordinates[a]
        lon2, lat2 = coordinates[b]
        plt.plot([lon1, lon2], [lat1, lat2], '-', zorder=1)
        plt.scatter(lon1, lat1, zorder=2)
        plt.text(lon1 + 0.5, lat1 + 0.5, cities[a][0], fontsize=8, zorder=3)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()
    

def plot_convergence():
    plt.figure(figsize=(9,4))
    plt.plot(history_best_dist,  label="best")
    plt.plot(history_mean_dist,  label="mean", alpha=0.7)
    plt.plot(history_median_dist,label="median", alpha=0.7)
    plt.title("Konvergencija GA")
    plt.xlabel("Generacija"); plt.ylabel("Dužina ture [km]")
    plt.grid(True); plt.legend(); plt.show()

#Kanonizuj startni grad (čisto za uporedive grafove/izveštaj)
#Prije ispisivanja/plotovanja rotiraj rutu da počinje, npr. u „Seattle“-u:
def canonical_start(route, start_idx=0):
    i = route.index(start_idx)
    r = route[i:]+route[:i]
    # simetrija (obrnut smer) je ista tura; opcionalno biraj “lepši”:
    return r if r[1] < r[-1] else [r[0]] + r[:0:-1]

def write_pergen_csv(path, best_hist, mean_hist, median_hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_km", "mean_km", "median_km"])
        for g, (b, m, d) in enumerate(zip(best_hist, mean_hist, median_hist), start=1):
            w.writerow([g, round(b, 4), round(m, 4), round(d, 4)])

def summarize_params(ga, extra=None):
    info = {
        "algo": "GA+ERX+inversion+2opt",
        "cities": len(cities),
        "sol_per_pop": ga.sol_per_pop,
        "num_parents_mating": ga.num_parents_mating,
        "mutation_type": ga.mutation_type,
        "mutation_percent_genes": ga.mutation_percent_genes,
        "selection": ga.parent_selection_type,
        "K_tournament": getattr(ga, "K_tournament", None),
        "allow_duplicate_genes": getattr(ga, "allow_duplicate_genes", None),
    }
    if extra:
        info.update(extra)
    return info

def to_py(x):
    if isinstance(x, (np.integer,)):     return int(x)
    if isinstance(x, (np.floating,)):    return float(x)
    if isinstance(x, (np.ndarray,)):     return x.tolist()
    if isinstance(x, (list, tuple)):     return [to_py(i) for i in x]
    if isinstance(x, dict):              return {str(k): to_py(v) for k, v in x.items()}
    return x

def write_summary_csv(path, summary_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in summary_dict.items():
            vv = to_py(v)  # <<< ključno
            # ako je list/dict, upiši kao JSON string radi preglednosti
            if isinstance(vv, (list, dict)):
                vv = json.dumps(vv, ensure_ascii=False)
            w.writerow([k, vv])
            

if __name__ == "__main__":
    num_genes = len(cities)
    pop_size  = 80

    #Reproduktivnost
    np.random.seed(42)



    # Inicijalna populacija: NN (6 kom) + random
    nn_count = 6
    nn_starts = np.linspace(0, num_genes-1, nn_count, dtype=int)
    # (opciono) 2-opt nad NN turama:
    nn_pop = [np.array(nn_route(distance_matrix, s), dtype=int) for s in nn_starts]
    #nn_pop = [np.array(two_opt_fast(r.tolist(), distance_matrix, tries_without_improve=60), dtype=int) for r in nn_pop]

    rand_count = pop_size - len(nn_pop)
    rand_pop = [np.random.permutation(num_genes) for _ in range(rand_count)]
    initial_population = np.array(nn_pop + rand_pop, dtype=int)

    # Crtanje lokacija
    plot_cities()

    # brisanje istorije iz prethodnih pokretanja
    history_best_dist.clear()
    history_mean_dist.clear()
    history_median_dist.clear()

    
    # Pokretanje GA
    ga = pygad.GA(
        num_generations=500,
        sol_per_pop=pop_size,
        num_parents_mating=20,
        initial_population=initial_population,
        fitness_func=fitness_func,
        num_genes=num_genes,
        gene_type=int,
        
        
        # KLJUČNO za permutacije:
        gene_space=list(range(num_genes)), # dozvoljene vrednosti gena
        allow_duplicate_genes=False,       # bez duplikata u jedinci
        
        #mutacija:         
        mutation_type="inversion",     # radi sjajno na TSP
        #mutation_type="swap",
        mutation_percent_genes=8,
        
        
        #crossover_type=ordered_crossover,
        crossover_type=erx_crossover,
        on_generation=on_generation,
        
        # kriterijumi za zaustavljanje:
        stop_criteria=["saturate_30"],
        
        # elitizam:
        keep_parents=2,
        #selekcija:
        parent_selection_type="tournament",    # ili "rank" kao alternativa
        K_tournament=3,  
    )
        
    ga.run()
    
    #tajmer
    t0 = time.perf_counter()
    ga.run()
    t1 = time.perf_counter()
    runtime_sec = t1 - t0


    # Konvergencija
    plot_convergence()

    # Najbolje rešenje
    solution, fitness, _ = ga.best_solution()
    best_route = fix_route(solution)
    best_route = two_opt_fast(best_route, distance_matrix, tries_without_improve=300) 
    best_route = canonical_start(best_route, start_idx=0) 
    dist, fuel = evaluate_route(best_route)

    print("\nNajbolja ruta (indeksi):", best_route)
    print("Ukupna distanca:        ", round(dist, 2), "km")
    print("Ukupna potrošnja goriva:", round(fuel, 2), "l")

    # Vizualizacija rute
    plot_route(best_route, title="Najbolja pronađena ruta")
    
    
    # >>> OVO DODAJ: snimanje CSV-ova
    exp_label = "GA_ERX_inversion_2opt"
    run_id    = 1      # promeni po potrebi
    seed_lbl  = 42     # ako koristiš random_seed i np.seed, zadrži isto

    pergen_path  = f"results/{exp_label}_seed{seed_lbl}_run{run_id}_pergen.csv"
    summary_path = f"results/{exp_label}_seed{seed_lbl}_run{run_id}_summary.csv"

    write_pergen_csv(pergen_path, history_best_dist, history_mean_dist, history_median_dist)

    summary = summarize_params(ga, extra={
        "distance_km": round(dist, 4),
        "fuel_l": round(fuel, 4),
        "generations_completed": int(ga.generations_completed),
        "runtime_sec": round(runtime_sec, 4),
        "start_city": cities[0][0],
        "fuel_per_km": fuel_per_km,
        "route_indices": best_route,     # snimamo i rutu (korisno za proveru)
        "route_city_names": [cities[i][0] for i in best_route],
    })
    write_summary_csv(summary_path, summary)

    print(f"\nCSV sa generacijama: {pergen_path}")
    print(f"CSV sa sažetkom:     {summary_path}")
