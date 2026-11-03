from pathlib import Path
import pandas as pd
import argparse
import json

config = {
    "geo": {"including_types": ["Point"], "Point": {"venue_category_id": "enum", "venue_category_name": "enum"}},
    "usr": {"properties": {}},
    "dyna": {"including_types": ["trajectory"], "trajectory": {"entity_id": "usr_id", "location": "geo_id"}},
    "info": {"distance_upper": 30.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Bigscity data to LibCity format")
    parser.add_argument("--city", type=str, required=True, help="City name")
    parser.add_argument("--path", type=Path, default=Path("data"), help="Path to the data directory")
    parser.add_argument("--output_path", type=Path, default=Path("raw_data"), help="Path to the output directory")
    return parser.parse_args()


def create_dyna(city, path, output_path):
    output_dir = output_path / f"std_{city}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path / city / f"{city}_checkins.csv")
    dyna = df[["trail_id", "user_id", "venue_id", "timestamp"]]
    dyna["timestamp"] = pd.to_datetime(dyna["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dyna["type"] = ["trajectory"] * len(dyna)
    # hash trail_id to create dyna_id int
    trails = sorted(dyna["trail_id"].unique())
    trail_to_id = {trail: i for i, trail in enumerate(trails)}
    # hash user_id to create entity_id int
    users = sorted(dyna["user_id"].unique())
    user_to_id = {user: i for i, user in enumerate(users)}
    dyna["entity_id"] = dyna["user_id"].map(user_to_id)
    # hash venue_id to create location int
    venues = sorted(dyna["venue_id"].unique())
    venue_to_id = {venue: i for i, venue in enumerate(venues)}
    dyna["location"] = dyna["venue_id"].map(venue_to_id)
    dyna["dyna_id"] = dyna["trail_id"].map(trail_to_id)
    dyna = dyna.rename(columns={"timestamp": "time"})
    dyna = dyna.reindex(columns=["dyna_id", "type", "time", "entity_id", "location"])
    dyna.to_csv(output_dir / f"std_{city}.dyna", index=False)

    poi = df[["venue_id", "venue_city_longitude", "venue_city_latitude", "venue_category_id", "venue_category"]]
    poi["coordinates"] = poi.apply(lambda x: f"[{x['venue_city_longitude']},{x['venue_city_latitude']}]", axis=1)
    poi["geo_id"] = poi["venue_id"].map(venue_to_id)
    poi["type"] = ["Point"] * len(poi)
    poi = poi.rename(columns={"venue_category": "venue_category_name"})
    poi = poi.reindex(columns=["geo_id", "type", "coordinates", "venue_category_id", "venue_category_name"])
    poi = poi.drop_duplicates()
    poi.to_csv(output_dir / f"std_{city}.geo", index=False)

    with open(output_dir / f"config.json", "w") as f:
        json.dump(config, f, indent=4)


def main(args):
    create_dyna(args.city, args.path, args.output_path)


if __name__ == "__main__":
    main(parse_args())
