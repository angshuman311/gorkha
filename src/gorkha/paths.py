from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = ROOT / "results"

# UTM zone 45N. All distance and area calculations use this CRS.
CRS_METRIC = "EPSG:32645"

# USGS origin for the 2015-04-25 Mw 7.8 event (longitude, latitude).
EPICENTER_LONLAT = (84.731, 28.231)

# Bounding box of the 11 survey districts with a margin (south, west, north, east).
STUDY_BBOX = (26.8, 84.3, 28.9, 86.7)

DISTRICTS = [
    "Gorkha",
    "Dhading",
    "Nuwakot",
    "Rasuwa",
    "Sindhupalchok",
    "Dolakha",
    "Ramechhap",
    "Okhaldhunga",
    "Makwanpur",
    "Sindhuli",
    "Kavrepalanchok",
]
