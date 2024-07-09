# load 2024_calendar.csv and plot the coordinates on a map

import pandas as pd
import folium
import numpy as np


def arc_distance(lat1, lon1, lat2, lon2):
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


# Load the data
""" example data
Grand Prix, Circuit, Date, Longitude, Latitude
Bahrain Grand Prix, Bahrain International Circuit, Mar 2, 26.0325°N, 50.5106°E
Saudi Arabian Grand Prix, Jeddah Corniche Circuit, Mar 9, 21.6319°N, 39.1044°E
"""
data = pd.read_csv("2024_calendar.csv")

# Parse the latitude and longitude
data["Longitude"], data["Latitude"] = zip(
    *data.apply(lambda x: parse_lat_lon(x["Longitude"], x["Latitude"]), axis=1)
)

# Create a map centered at 0, 0
m = folium.Map(location=[0, 0], zoom_start=2, prefer_canvas=True, control_scale=True)
folium.TileLayer("cartodb dark_matter").add_to(m)

# Add markers for each row in the data
for i, row in data.iterrows():
    folium.Marker(
        location=[row["Longitude"], row["Latitude"]],
        icon=folium.DivIcon(
            html=f"<div style='font-weight: bold; background: #999; padding: 5px; border-radius: 5px; text-align: center; width: 100px; border: 1px solid #000; position: absolute; transform: translate(-50%, -50%); top: 50%; left: 50%;'>{i+1}: {row['Grand Prix']}</div>"
        ),
    ).add_to(m)

    # Add lines connecting the points with a text in the center of the line showing the distance between the points
    if i == 0:
        continue

    lat1, lon1 = data.loc[i - 1, ["Latitude", "Longitude"]]
    lat2, lon2 = data.loc[i, ["Latitude", "Longitude"]]
    folium.PolyLine(
        locations=[[lon1, lat1], [lon2, lat2]],
        color="red",
        weight=2,
        opacity=0.5,
        dash_array="5, 5",
    ).add_to(m)

    # Add text in the middle of the line
    folium.Marker(
        location=[(lon1 + lon2) / 2, (lat1 + lat2) / 2],
        icon=folium.DivIcon(
            html=f"<div style='font-weight: bold; font-color: #ccc; background: #666; padding: 5px; border-radius: 5px; text-align: center; width: 100px; border: 1px solid #000; position: absolute; transform: translate(-50%, -50%); top: 50%; left: 50%;'>{arc_distance(lat1, lon1, lat2, lon2):.2f} km</div>"
        ),
    ).add_to(m)

# Save the map
m.save("2024_calendar.html")
