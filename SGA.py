import random
import time

import numpy
import numpy.random as npr
from sql_manager import SqlManager


def one_max(data):
    return sum(data)


def peak(data):
    return numpy.prod(data)


def trap(data: str):
    return 3 * len(data) * peak(data) - one_max(data)


class SGA:
    def __init__(self, population_size, problem_size, fitness_function, max_gen):
        self.population_size = int(population_size / 2) * 2
        self.problem_size = problem_size
        self.fitness_function = fitness_function
        self.population = self.__init_population()
        self.max_gen = max_gen
        self.generation = 1
        self.best_chromosome = [1 for x in range(self.problem_size)]
        self.selection_problems_probability_list = []

    def __init_population(self):
        population = []
        for index1 in range(self.population_size):
            population.append([random.choice([0, 1]) for index2 in range(self.problem_size)]),
        return population

    def terminate(self):
        if self.generation >= self.max_gen:
            return True
        if self.best_chromosome in self.population:
            return True
        return False

    def update_selection_problems_probability_list(self):
        sum_fitness = sum([self.fitness_function(c) for c in self.population])
        if sum_fitness != 0:
            self.selection_problems_probability_list = [self.fitness_function(c) / sum_fitness for c in self.population]
        else:
            self.selection_problems_probability_list = [1 / len(self.population) for c in range(self.population_size)]

    def fps(self):
        # return an index
        return npr.choice(self.population_size, p=self.selection_problems_probability_list)

    def single_point_cross_over(self, chromosome1, chromosome2, p=0.8):

        if random.random() > p:
            return chromosome1, chromosome2
        else:
            point = random.randint(0, self.problem_size - 1)
            new_chromosome1 = chromosome1[0:point]
            new_chromosome1.extend(chromosome2[point: self.problem_size])
            new_chromosome2 = chromosome2[0:point]
            new_chromosome2.extend(chromosome1[point: self.problem_size])

            return new_chromosome1, new_chromosome2

    def shuffle_and_pair_parents(self, parents_pool):
        random.shuffle(parents_pool)
        parents_pairs = [(parents_pool[i], parents_pool[i + 1]) for i in range(0, self.population_size, 2)]
        return parents_pairs

    def bit_flipping(self, chromosome, p=None):
        if p is None:
            p = 1 / self.problem_size

        for gen_index in range(0, self.problem_size):
            if random.random() <= p:
                chromosome[gen_index] = 0 if chromosome[gen_index] == 1 else 1

        return chromosome

    def generate_offsprings(self, parents_pool):
        off_springs = []
        for i in range(0, self.population_size, 2):
            offspring_pair = self.single_point_cross_over(parents_pool[i], parents_pool[i + 1], p=0.8)
            off_springs.append(self.bit_flipping(offspring_pair[0]))
            off_springs.append(self.bit_flipping(offspring_pair[1]))
        return off_springs

    def run(self):

        while not self.terminate():
            # if self.generation % 20 == 0:
            #     print("|", end="")
            self.update_selection_problems_probability_list()
            parents_pool = [self.population[self.fps()] for i in range(self.population_size)]
            random.shuffle(parents_pool)
            offsprings = self.generate_offsprings2(parents_pool=parents_pool)
            self.population = offsprings
            self.generation += 1
        return max(self.population, key=lambda item: self.fitness_function(item))


if __name__ == '__main__':
    time1 = time.time()
    sql_manager = SqlManager(file="information.sqlite")
    sql_manager.create_database()
    for function in [one_max, peak, trap]:
        for max_gen in [100, 200, 300]:
            for problem_size in [10, 30, 50, 70, 100]:
                for pop_size in [50, 100, 200, 300]:
                    for i in range(10):
                        print("\n_________________________________________________")
                        print(
                            f"problem_size={problem_size}\npop_size={pop_size}\nmax_gen={max_gen}\nfitness ={function.__name__}")
                        sga = SGA(population_size=pop_size, problem_size=problem_size, fitness_function=function,
                                  max_gen=max_gen)
                        result = sga.run()
                        sql_manager.add_row(fitness=function.__name__, max_gen=max_gen, problem_size=problem_size,
                                            pop_size=pop_size, result="".join([str(i) for i in result]),
                                            generation=sga.generation)

    print("time : ", time.time() - time1)
