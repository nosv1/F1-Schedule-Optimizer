# https://www.youtube.com/watch?v=j0Vc9vK4Qlo&t=1s

import json
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

from GeneticAlgorithm.GeneticAlgorithm import GeneticAlgorithm
from GrandPrix import GrandPrix
from Schedule import Schedule

print("-" * 50)
print(f"Loading 2024 distance matrix...", end=" ")
_2024_CALENDARS_PATH: str = r"calendars\2024\defaults"
_2024_DISTANCE_MATRIX = json.load(
    open(rf"{_2024_CALENDARS_PATH}\2024_distance_matrix.json", "r")
)
print(f"done.")

print(f"Reading 2024 calendar and sub-calendars...", end=" ")
_2024_CALENDAR: pd.DataFrame = pd.read_csv(rf"{_2024_CALENDARS_PATH}\2024_calendar.csv")
_2024_CALENDAR_AMERICAS: pd.DataFrame = pd.read_csv(
    rf"{_2024_CALENDARS_PATH}\2024_calendar_americas.csv"
)
_2024_CALENDAR_ASIA: pd.DataFrame = pd.read_csv(
    rf"{_2024_CALENDARS_PATH}\2024_calendar_asia.csv"
)
_2024_CALENDAR_EUROPE: pd.DataFrame = pd.read_csv(
    rf"{_2024_CALENDARS_PATH}\2024_calendar_europe.csv"
)
print(f"done.")

print(f"Combining calendars...", end=" ")
CALENDARS = []
# CALENDARS.append(_2024_CALENDAR_ASIA)
CALENDARS.append(_2024_CALENDAR_EUROPE)
# CALENDARS.append(_2024_CALENDAR_AMERICAS)
# CALENDARS.append(_2024_CALENDAR)
assert CALENDARS != []
CALENDAR = pd.concat(CALENDARS, ignore_index=True)
print(f"done.")

AUSTRALIA_GP_NAME = "Australian"
IS_AUSTRALIA_IN_CALENDAR = AUSTRALIA_GP_NAME in CALENDAR["Grand Prix"].values
DO_AUSTRALIA_POS_CHECK = True and IS_AUSTRALIA_IN_CALENDAR
AUSTRALIA_POSITION = 0

ABU_DHABI_GP_NAME = "Abu Dhabi"
IS_ABU_DHABI_IN_CALENDAR = ABU_DHABI_GP_NAME in CALENDAR["Grand Prix"].values
DO_ABU_DHABI_POS_CHECK = False and IS_ABU_DHABI_IN_CALENDAR
ABU_DHABI_POSITION = -1

US_RACES = ["United States", "Las Vegas", "Miami"]


def swap_grand_prix(
    grand_prix: list[GrandPrix], gp_name: str, position: int
) -> list[GrandPrix]:
    if grand_prix[position].name != gp_name:
        for i in range(len(grand_prix)):
            if grand_prix[i].name == gp_name:
                grand_prix[position], grand_prix[i] = (
                    grand_prix[i],
                    grand_prix[position],
                )
                break

    return grand_prix


def evaluate_schedule(schedule: Schedule) -> Schedule:
    schedule.fitness = -schedule.total_distance

    return schedule


def generate_schedules(count: int, grand_prix: list[GrandPrix]) -> list[Schedule]:
    schedules = []

    while len(schedules) < count:
        grand_prix = random.sample(grand_prix, len(grand_prix))
        if DO_AUSTRALIA_POS_CHECK:
            grand_prix = swap_grand_prix(
                grand_prix, AUSTRALIA_GP_NAME, AUSTRALIA_POSITION
            )

        if DO_ABU_DHABI_POS_CHECK:
            grand_prix = swap_grand_prix(
                grand_prix, ABU_DHABI_GP_NAME, ABU_DHABI_POSITION
            )

        schedule = Schedule(
            grand_prix=grand_prix,
            distance_matrix=_2024_DISTANCE_MATRIX,
        )

        schedules.append(schedule)
    return schedules


def mutate_schedules(chromosomes: list[Schedule], mutation_rate: float) -> Schedule:
    chromosomes = sorted(chromosomes, key=lambda x: x.fitness, reverse=True)

    next_generation = []
    num_breeding_parents = 4
    for i in range(0, len(chromosomes), num_breeding_parents):
        parents = chromosomes[i : i + num_breeding_parents]

        # perfectly zip the two parents together, then loop back through the embryo randomly removing one of the genes, ensuring all grand prix are kept, but there are also no duplicates
        child: list[GrandPrix] = []
        genes_selected: set[GrandPrix] = set()
        gene_pool: list[GrandPrix] = []
        for genes in zip(*[p.genes for p in parents]):
            genes: tuple[GrandPrix]

            gene = genes[random.randint(0, len(genes) - 1)]
            if gene not in genes_selected:
                genes_selected.add(gene)
                child.append(gene)

            for g in genes:
                if g != gene:
                    gene_pool.append(g)

        for gene in gene_pool:
            if gene not in genes_selected:
                genes_selected.add(gene)
                child.append(gene)

        shifted_child = []
        start_index = random.randint(0, len(child) - 1)
        chunk_size = random.randint(1, (len(child) / 2) - 1)
        chunk = []
        i = start_index
        chunk_filled = lambda: len(chunk) == chunk_size
        while not chunk_filled() or len(shifted_child) + len(chunk) < len(child):
            if not chunk_filled() and (i >= start_index or chunk != []):
                chunk.append(child[i])
            else:
                shifted_child.append(child[i])
            i += 1
            if i >= len(child):
                i = 0
        shifted_child += chunk

        if DO_AUSTRALIA_POS_CHECK:
            child = swap_grand_prix(
                shifted_child, AUSTRALIA_GP_NAME, AUSTRALIA_POSITION
            )

        if DO_ABU_DHABI_POS_CHECK:
            child = swap_grand_prix(
                shifted_child, ABU_DHABI_GP_NAME, ABU_DHABI_POSITION
            )

        # CHILD
        next_generation.append(
            Schedule(grand_prix=list(child), distance_matrix=_2024_DISTANCE_MATRIX)
        )
        # KEEP PARENT or CREATE NEW RANDOM SCHEDULE
        for _ in range(num_breeding_parents - 1):
            if random.random() < mutation_rate:
                next_generation.append(generate_schedules(1, chromosomes[0].genes)[0])
            else:
                next_generation.append(random.choice(parents))

    return next_generation


if __name__ == "__main__":
    grand_prix: list[GrandPrix] = [GrandPrix(row) for _, row in CALENDAR.iterrows()]

    # population size is based on how many permutations there are of the grand prix-ish
    population_size = int(
        math.factorial(len(grand_prix)) ** (1 / (len(grand_prix) / 4))
    )
    mutation_rate = 0.1
    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate,
        keep_best=True,
        chromosome_generator=generate_schedules,
        chromosome_evaluator=evaluate_schedule,
        chromosome_displayer=None,
        chromosome_mutator=mutate_schedules,
        chromosome_writer=None,
    )
    print("-" * 50)
    print(f"GA initialized with:")
    print(f"\tPopulation size: {population_size}")
    print(f"\tMutation rate: {mutation_rate}")
    print("-" * 50)

    print("Initializing population...", end=" ")
    ga.initialize_population(count=ga.population_size, grand_prix=grand_prix)
    print("done.")

    input("Press Enter to start...")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    ax1 = ax.twinx()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax1.set_ylabel("Standard Deviation")

    ax2.set_xlabel("Population")
    ax2.set_ylabel("Fitness")

    generations = 0
    start_time = time.time()
    while True:
        generations += 1

        ga.evaluate_chromosomes()

        print(f"Generation {generations}:")
        print(f"Average fitness: {ga.average_fitness}")
        print(f"Standard deviation: {ga.standard_deviation}")
        print(f"Fittest chromosome: {ga.fittest_chromosome.fitness}")
        print(f"Elapsed time: {time.time() - start_time:.2f}s")
        start_time = time.time()

        ax.plot(generations, ga.fittest_chromosome.fitness, "ro")
        ax.plot(generations, ga.average_fitness, "bo")
        ax1.plot(generations, ga.standard_deviation, "go")
        ax.legend(["Fittest Chromosome", "Average Fitness"])
        ax1.legend(["Standard Deviation"])

        if generations % 10 == 0 or generations == 1:
            best_schedule: Schedule = ga.fittest_chromosome
            best_schedule.create_map()
            ax2.clear()
            fitnesses = sorted([c.fitness for c in ga.population], reverse=True)
            ax2.plot(fitnesses, "bo")

        ga.calculate_weights()
        ga.create_offspring()

        plt.pause(0.05)

        if ga.standard_deviation == 0:
            break

    pass
