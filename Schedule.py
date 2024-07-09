import folium
import matplotlib.cm as cm
from typing import Optional
from utils import arc_distance

from GeneticAlgorithm.GeneticAlgorithm import Chromosome
from GrandPrix import GrandPrix


class Schedule(Chromosome):
    def __init__(
        self, grand_prix: list[GrandPrix], distance_matrix: dict[str, dict[str, float]]
    ) -> None:
        super().__init__(grand_prix)
        self.genes: list[GrandPrix]
        self.total_distance: float = self.set_total_distance(
            self.genes, distance_matrix
        )

    @staticmethod
    def set_total_distance(
        grand_prix: list[GrandPrix], distance_matrix: dict[str, dict[str, float]]
    ) -> float:
        total_distance = 0
        for i in range(len(grand_prix) - 1):
            total_distance += grand_prix[i].distance_to(
                grand_prix[i + 1], distance_matrix
            )
        return total_distance

    @property
    def grand_prix(self) -> list[GrandPrix]:
        return self.genes

    def create_map(self, name: Optional[str] = None) -> None:
        m = folium.Map(
            location=[0, 0], zoom_start=3, prefer_canvas=True, control_scale=True
        )
        folium.TileLayer("cartodb dark_matter").add_to(m)

        # Add markers for each row in the data
        for i, grand_prix in enumerate(self.grand_prix):

            # Calculate the color gradient based on the index number
            color = cm.Blues(i / len(self.grand_prix))
            color_as_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            text_color = "#000" if i < len(self.grand_prix) / 2 else "#fff"

            folium.Marker(
                location=[grand_prix.longitude, grand_prix.latitude],
                icon=folium.DivIcon(
                    html=f"<div style='font-weight: bold; color: {text_color}; background: {color_as_hex}; padding: 5px; border-radius: 5px; text-align: center; width: 100px; border: 1px solid #000; position: absolute; transform: translate(-50%, -50%); top: 50%; left: 50%;'>{i+1}: {grand_prix.name}</div>"
                ),
            ).add_to(m)

            # Add lines connecting the points with a text in the center of the line showing the distance between the points
            if i == 0:
                continue

            lat1, lon1 = (
                self.grand_prix[i - 1].latitude,
                self.grand_prix[i - 1].longitude,
            )
            lat2, lon2 = self.grand_prix[i].latitude, self.grand_prix[i].longitude
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

        m.save(rf"maps\{name if name else 'schedule'}.html")
