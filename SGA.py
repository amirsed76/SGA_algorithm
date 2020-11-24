import random
import numpy
import numpy.random as npr
from sql_manager import SqlManager


def one_max(data: str):
    return sum([int(ch) for ch in data])


def peak(data: str):
    return numpy.prod([int(ch) for ch in data])


def trap(data: str):
    return 3 * len(data) * peak(data) - one_max(data)


class Chromosome:
    def __init__(self, data, fitness_function):
        self.data = data
        self.fitness_function = fitness_function

    @property
    def fitness(self):
        return self.fitness_function(self.data)

    def __str__(self):
        return self.data


class SGA:
    def __init__(self, population_size, problem_size, fitness_function, max_gen):
        self.population_size = int(population_size / 2) * 2
        self.problem_size = problem_size
        self.fitness_function = fitness_function
        self.population = self.__init_population()
        self.max_gen = max_gen
        self.generation = 1

    def __init_population(self):
        population = []
        for index1 in range(self.population_size):
            chromosome = Chromosome(data=''.join((random.choice("10") for index2 in range(self.problem_size))),
                                    fitness_function=self.fitness_function)
            population.append(chromosome)
        return population

    def terminate(self):
        if self.generation >= self.max_gen:
            return True
        for chromosome in self.population:
            if chromosome.data == "".join("1" for i in range(self.problem_size)):
                return True
        return False

    def fps(self):
        # return an index
        sum_fitness = sum([c.fitness for c in self.population])
        if sum_fitness != 0:
            selection_problems = [c.fitness / sum_fitness for c in self.population]
        else:
            selection_problems = [1 / len(self.population) for c in self.population]
        return npr.choice(len(self.population), p=selection_problems)

    def single_point_cross_over(self, chromosome1, chromosome2, p=0.8):

        if random.random() > p:
            return chromosome1, chromosome2
        else:
            point = random.randint(0, self.problem_size - 1)
            new_chromosome1 = Chromosome(
                data=chromosome1.data[0:point] + chromosome2.data[point: self.problem_size],
                fitness_function=self.fitness_function)

            new_chromosome2 = Chromosome(
                data=chromosome2.data[0:point] + chromosome1.data[point: self.problem_size],
                fitness_function=self.fitness_function)

            return new_chromosome1, new_chromosome2

    def shuffle_and_pair_parents(self, parents_pool):
        random.shuffle(parents_pool)
        parents_pairs = [(parents_pool[i], parents_pool[i + 1]) for i in range(0, self.population_size, 2)]
        return parents_pairs

    def bit_flipping(self, chromosome, p=None):
        if p is None:
            p = 1 / self.problem_size

        data = ""
        for gen_index in range(0, self.problem_size):
            if random.random() <= p:
                data += "0" if chromosome.data[gen_index] == "1" else "1"
            else:
                data += chromosome.data[gen_index]
        chromosome.data = data

    def generate_offsprings(self, parents_pairs):
        off_springs = []
        for parent_pair in parents_pairs:
            offspring_pair = self.single_point_cross_over(parent_pair[0], parent_pair[1], p=0.8)
            self.bit_flipping(offspring_pair[0])
            self.bit_flipping(offspring_pair[1])
            off_springs.append(offspring_pair[0])
            off_springs.append(offspring_pair[1])
        return off_springs

    def run(self):
        while not self.terminate():
            if self.generation % 20 == 0:
                print("|", end="")
            parents_pool = [self.population.pop(self.fps()) for i in range(self.population_size)]
            parents_pairs = self.shuffle_and_pair_parents(parents_pool=parents_pool)
            offsprings = self.generate_offsprings(parents_pairs=parents_pairs)
            self.population = offsprings
            self.generation += 1
        return max(self.population, key=lambda item: item.fitness)


if __name__ == '__main__':
    sql_manager = SqlManager(file="information.sqlite")
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
                                            pop_size=pop_size, result=result, generation=sga.generation)
