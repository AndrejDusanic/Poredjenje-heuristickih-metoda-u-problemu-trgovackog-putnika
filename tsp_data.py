import numpy as np
import itertools
from math import radians, sin, cos, sqrt, atan2

# Podaci
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

# za crtanje (x=lon, y=lat)
coordinates = np.array([(lon, lat) for _, lat, lon in cities])
latlons     = np.array([(lat, lon) for _, lat, lon in cities])

def haversine(c1, c2):
    R = 6371.0
    lat1, lon1 = radians(c1[0]), radians(c1[1])
    lat2, lon2 = radians(c2[0]), radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def calculate_distance_matrix(latlons):
    n = len(latlons)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = haversine(latlons[i], latlons[j])
        D[i, j] = D[j, i] = d
    return D

distance_matrix = calculate_distance_matrix(latlons)
fuel_per_km = 1.2
