import numpy as np
import pandas as pd
import json


def arc_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


def parse_lat_lon(longitude: str, latitude: str):
    # if the last character is W or S, then it is negative
    lon = float(longitude[:-2])
    lat = float(latitude[:-2])
    if longitude[-1] == "S":
        lon = -lon
    if latitude[-1] == "W":
        lat = -lat
    return lon, lat


def read_data(filename):
    data = pd.read_csv(filename)
    data["Longitude"], data["Latitude"] = zip(
        *data.apply(lambda x: parse_lat_lon(x["Longitude"], x["Latitude"]), axis=1)
    )
    return data


if __name__ == "__main__":
    data = read_data("2024_calendar.csv")

    # Calculate distance matrix
    dist_matrix = {}
    for _, row in data.iterrows():
        grand_prix = row["Grand Prix"]
        dist_matrix[grand_prix] = {}
        for _, row2 in data.iterrows():
            grand_prix2 = row2["Grand Prix"]
            dist_matrix[grand_prix][grand_prix2] = arc_distance(
                row["Latitude"], row["Longitude"], row2["Latitude"], row2["Longitude"]
            )

    # Write distance matrix to JSON file
    with open("2024_distance_matrix.json", "w") as f:
        json.dump(dist_matrix, f, indent=4)

    pass
