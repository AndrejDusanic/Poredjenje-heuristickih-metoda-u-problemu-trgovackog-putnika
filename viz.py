import matplotlib.pyplot as plt
from tsp_data import cities, coordinates

def plot_cities():
    plt.figure(figsize=(10, 6))
    for name, lat, lon in cities:
        plt.scatter(lon, lat, zorder=2)
        plt.text(lon + 0.5, lat + 0.5, name, fontsize=8, zorder=3)
    plt.title("Lokacije gradova")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.grid(True); plt.show()

def plot_route(route, title="Najbolja pronađena ruta"):
    plt.figure(figsize=(10, 6))
    n = len(route)
    for i in range(n):
        a, b = route[i], route[(i + 1) % n]
        lon1, lat1 = coordinates[a]
        lon2, lat2 = coordinates[b]
        plt.plot([lon1, lon2], [lat1, lat2], '-', zorder=1)
        plt.scatter(lon1, lat1, zorder=2)
        plt.text(lon1 + 0.5, lat1 + 0.5, cities[a][0], fontsize=8, zorder=3)
    plt.title(title)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.grid(True); plt.show()

def plot_convergence(best_hist, mean_hist=None, median_hist=None):
    plt.figure(figsize=(9, 4))
    plt.plot(best_hist, label="best")
    if mean_hist is not None:
        plt.plot(mean_hist, label="mean", alpha=0.7)
    if median_hist is not None:
        plt.plot(median_hist, label="median", alpha=0.7)
    plt.title("Konvergencija")
    plt.xlabel("Generacija"); plt.ylabel("Dužina ture [km]")
    plt.grid(True); plt.legend(); plt.show()
