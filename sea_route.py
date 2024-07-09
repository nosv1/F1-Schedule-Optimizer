import searoute as sr
import folium

origin = [0.414309, 50.607101]  # lon, lat
destination = [-97.015268, 29.107776]  # lon, lat

route = sr.searoute(origin, destination)

coords = route["geometry"]["coordinates"]  # [[lat, lon], [lat, lon], ...]

m = folium.Map(location=[0, 0], zoom_start=3, prefer_canvas=True, control_scale=True)
folium.PolyLine([[lat, lon] for lon, lat in coords], color="blue").add_to(m)
folium.TileLayer("cartodb dark_matter").add_to(m)

# add markers for origin and destination
folium.Marker(location=origin[::-1], icon=folium.Icon(color="green")).add_to(m)
folium.Marker(location=destination[::-1], icon=folium.Icon(color="red")).add_to(m)


m.save("route.html")

pass
