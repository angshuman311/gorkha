# GORKHA

**Graph-based Operational Reconnaissance Kit for Human-in-the-loop Assessment**

GORKHA estimates building damage in all wards of an earthquake region from a small number
of field surveys, and tells the survey teams where to go next. It uses a graph of the wards,
the early data that are available without a visit (ShakeMap, terrain, satellite images), and
the road network with its blockages.

The test case is the 2015 Gorkha earthquake sequence in Nepal (945 wards, 11 districts).

Status: research code for a CS230 course project. The interfaces change without notice.

## Install

```bash
pip install gorkha
```

For the deep learning parts: `pip install "gorkha[deep]"`. For the Sentinel-1 processing at
the Alaska Satellite Facility: `pip install "gorkha[sar]"`.

## Data

The scripts download public data: the Nepal damage survey (NPC and Kathmandu Living Labs),
OpenStreetMap boundaries and roads (ODbL), USGS ShakeMaps, the NASA ARIA Damage Proxy Map,
the USGS landslide inventory, the Copernicus DEM, Sentinel-1 and Sentinel-2 images. The
Sentinel-1 coherence needs a free NASA Earthdata account.

## License

MIT. See LICENSE. The data sets have their own licenses (see the data section).
