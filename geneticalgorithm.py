from functools import total_ordering
import math
import random
import pandas as pd
import multiprocessing as mp


def mutate(individual):
    return individual.mutate()


@total_ordering
class Individual:
    def __init__(self, solution: pd.DataFrame, maximize: bool = True):
        self.solution = solution
        self.maximize = maximize
        self.__calculate_fitness()

    def __calculate_fitness(self) -> None:
        indexes = {}
        for row in self.solution.iterrows():
            for value in row[1]:
                name = indexes.setdefault(value, {})
                name.setdefault("first", row[0])
                name["last"] = row[0]
        durations = {
            name: index["last"] - index["first"] + 1 for name, index in indexes.items()
        }
        self.durations = pd.Series(durations)
        if self.maximize:
            self.fitness = 13 - self.durations.max()
        else:
            self.fitness = self.durations.max()

    def mate(self, other: object):
        indexes = random.sample(range(12), 6)
        child1solution = self.solution.copy()
        child2solution = other.solution.copy()
        child1solution.update(other.solution.loc[indexes])
        child2solution.update(self.solution.loc[indexes])
        return (
            Individual(child1solution, self.maximize),
            Individual(child2solution, self.maximize),
        )

    def mutate(self):
        indexes = random.sample(range(12), 4)
        childsolution = self.solution.copy()
        for x, y in zip(*[iter(indexes)] * 2):
            childsolution.loc[x], childsolution.loc[y] = (
                childsolution.loc[y],
                childsolution.loc[x],
            )

        return Individual(childsolution, self.maximize)

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "fitness"):
            return NotImplemented
        return self.fitness == other.fitness

    def __lt__(self, other: object) -> bool:
        if not hasattr(other, "fitness"):
            return NotImplemented
        return self.fitness < other.fitness

    def __str__(self):
        return self.solution.__str__()


class GeneticAlgorithm:
    def run(
        self,
        starting_solutions,
        maximize: bool = True,
        generations_unchanged: int = 40,
    ):
        self.pool = mp.Pool()
        population = pd.Series(
            self.pool.starmap(Individual, ((x, maximize) for x in starting_solutions))
        )

        score = (population.max() if maximize else population.min()).fitness
        unchanged_count = 1

        while unchanged_count <= generations_unchanged:
            print(
                f"{'Maximizing' if maximize else 'Minimizing'} fitness [{unchanged_count}/{generations_unchanged}]: {score}"
            )

            parents = self.pick_parents(population, math.floor(len(population) / 8) * 2)

            offspring = self.mutate_parents(parents)

            population = pd.concat((population, offspring), ignore_index=True)
            population = population.drop(
                population.map(lambda x: x.fitness).nsmallest(offspring.size).index
                if maximize
                else population.map(lambda x: x.fitness).nlargest(offspring.size).index
            ).reset_index(drop=True)

            if (
                new_score := (
                    population.max() if maximize else population.min()
                ).fitness
            ) == score:
                unchanged_count += 1
            else:
                unchanged_count = 1
                score = new_score

        self.pool.close()
        return population.max() if maximize else population.min()

    def pick_parents(self, population: pd.Series, n: int):
        return population.sample(
            n, weights=population.map(lambda x: 13 - x.fitness)
        ).reset_index(drop=True)

    def mate_parents(self, parents):
        offspring = []
        for x, y in zip(*[iter(parents)] * 2):
            offspring += x.mate(y)
        return pd.Series(offspring)

    def mutate_parents(self, parents):
        return pd.Series(self.pool.map(mutate, parents))
