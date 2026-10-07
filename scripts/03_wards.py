"""Build the district, municipality, and ward polygons of the study area."""

from gorkha import wards

if __name__ == "__main__":
    r = wards.build()
    print(r["summary"].to_string())
    print(f"\nmunicipalities: {r['n_municipalities']}, wards: {r['n_wards']}")
    print(f"agreement of the point test with the subarea links: {r['parent_agreement']:.4f}")
    print(f"wards with a parent from the subarea link only: {r['parent_from_subarea_only']}")
    print(f"wards without a ward number: {r['wards_without_number']}")
    print(f"wards where the name number and the `ward` tag differ: {r['ward_number_differs']}")
    print(f"duplicate (municipality, ward number) pairs: {r['duplicate_ward_numbers']}")
