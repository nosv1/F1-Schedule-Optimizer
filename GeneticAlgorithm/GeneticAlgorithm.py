from __future__ import annotations

from io import TextIOWrapper
from math import e
from statistics import mean, stdev
from random import randint, choices
from typing import Callable

from .Chromosome import Chromosome
from .Gene import Gene


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        keep_best: bool,
        chromosome_generator: Callable,
        chromosome_evaluator: Callable,
        chromosome_displayer: Callable,
        chromosome_mutator: Callable,
        chromosome_writer: Callable = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.keep_best = keep_best
        self.__chromosome_generator = chromosome_generator
        self.__chromosome_evaluator = chromosome_evaluator
        self.__chromosome_displayer = chromosome_displayer
        self.__chromosome_mutator = chromosome_mutator
        self.__chromosome_writer = chromosome_writer

        self.__population: list[Chromosome] = []
        self.__weights: list[float] = []
        self.__fitnesses: list[float] = []

        self.__average_fitness: float = None
        self.__standard_deviation: float = None
        self.__fittest_chromosome: Chromosome = None

    @property
    def average_fitness(self) -> float:
        return self.__average_fitness

    @property
    def standard_deviation(self) -> float:
        return self.__standard_deviation

    @property
    def fitnesses(self) -> list[float]:
        return self.__fitnesses

    @property
    def fittest_chromosome(self) -> Chromosome:
        return self.__fittest_chromosome

    @property
    def population(self) -> list[Chromosome]:
        return self.__population

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # generate chromosomes

    def generate_chromosomes(self, count: int = 1, **kwargs) -> None:
        return self.__chromosome_generator(count=count, **kwargs)

    def initialize_population(self, **kwargs) -> None:
        self.__population = self.generate_chromosomes(**kwargs)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # evaluate chromosomes

    def evaluate_chromosomes(self) -> None:
        """
        Assign a fitness value to each chromosome in the population.
        """
        self.__population = [self.__chromosome_evaluator(c) for c in self.__population]

        # calculate fittest chromosome, mean, and standard deviation
        self.__fitnesses: list[float] = []
        self.__fittest_chromosome: Chromosome = Chromosome(genes=[])
        self.__fittest_chromosome.fitness = float("-inf")
        for chromosome in self.__population:
            self.__fitnesses.append(chromosome.fitness)
            self.__fittest_chromosome = max(
                self.__fittest_chromosome, chromosome, key=lambda c: c.fitness
            )

        # self.__fitnesses = [c.fitness for c in self.__population]
        # self.__fittest_chromosome = max(self.__population, key=lambda c: c.fitness)
        self.__average_fitness = mean(self.fitnesses)
        self.__standard_deviation = stdev(self.fitnesses)

    def calculate_weights(self) -> None:
        """
        Calculates relative weights of the chromosomes.
        """
        # Note, the order of the distribution aligns with the population.
        max_fitness: float = float("-inf")
        min_fitness: float = float("inf")
        for chromosome in self.__population:
            max_fitness = max(max_fitness, chromosome.fitness)
            min_fitness = min(min_fitness, chromosome.fitness)

        try:
            self.__weights = [
                (c.fitness - min_fitness) / (max_fitness - min_fitness)
                for c in self.__population
            ]
        except ZeroDivisionError:
            self.__weights = [1.0 for _ in self.__population]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # create offspring

    def create_offspring(self) -> None:
        """
        Create a new population based on the chromosome weights to randomly
        select chromosomes, and then try to mutate them.
        """
        population_copy: list[Chromosome] = self.population.copy()
        self.__population = [
            chromosome
            for chromosome in choices(
                population=population_copy,
                weights=self.__weights,
                k=self.population_size,
            )
        ]
        self.__population = self.__chromosome_mutator(
            chromosomes=self.__population, mutation_rate=self.mutation_rate
        )
        if self.keep_best:
            self.__population[randint(0, self.population_size - 1)] = (
                self.fittest_chromosome
            )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # display chromosomes

    def display_chromosome(self, chromosome: Chromosome) -> None:
        self.__chromosome_displayer(chromosome)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # write chromosomes to file

    def save_chromosome(self, chromosome: Chromosome) -> None:
        self.__chromosome_writer(chromosome)
