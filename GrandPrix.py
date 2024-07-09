from __future__ import annotations

from datetime import datetime
import pandas as pd

from GeneticAlgorithm.GeneticAlgorithm import Gene
from utils import arc_distance, parse_lat_lon


class GrandPrix(Gene):
    """
    Grand Prix were parsed from csv file
    Grand Prix,Circuit,Date,Longitude,Latitude
    Bahrain,Bahrain International Circuit,Mar 2,26.0325°N,50.5106°E
    Saudi Arabian,Jeddah Corniche Circuit,Mar 9,21.6319°N,39.1044°E
    ...
    """

    def __init__(self, csv_row: pd.Series) -> None:
        self.name = csv_row["Grand Prix"]
        self.circuit = csv_row["Circuit"]
        self.original_date = datetime.strptime(csv_row["Date"], "%b %d")
        self.longitude, self.latitude = parse_lat_lon(
            csv_row["Longitude"], csv_row["Latitude"]
        )
        super().__init__()

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: GrandPrix) -> bool:
        return self.name == other.name

    def distance_to(
        self, other: GrandPrix, distance_matrix: dict[str, dict[str, float]]
    ) -> float:
        return distance_matrix[self.name][other.name]
