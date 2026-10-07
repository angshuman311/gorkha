"""Download the raw data. Usage: python scripts/01_download.py [source ...]"""

import sys

from gorkha import download

SOURCES = {
    "survey": download.survey,
    "shakemap": download.shakemap,
    "shakemap_variants": download.shakemap_variants,
    "admin": download.admin_boundaries,
    "roads": download.roads,
    "roads_full": download.roads_full,
    "dem": download.dem,
    "landslides": download.landslides,
    "gdif": download.gdif,
    "dpm": download.dpm,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    failed = []
    for name in names:
        print(f"== {name}")
        try:
            SOURCES[name]()
        except Exception as e:
            failed.append(name)
            print(f"FAILED {name}: {e}")
    if failed:
        sys.exit(f"failed sources: {', '.join(failed)}")
