import random
from pprint import pprint


class SGA:
    def __init__(self, population_length, solve_length, fitness_function, solve_function):
        self.population_length = population_length
        self.solve_length = solve_length
        self.population = self.init_population(population_length, solve_length)
        self.fitness_function = fitness_function
        self.solve_function = solve_function

    def init_population(self, population_length, solve_length):
        result = []
        for i in range(population_length):
            result.append(self.make_chromosome(solve_length))
        return result

    @staticmethod
    def make_chromosome(solve_length):
        result_str = ''.join((random.choice("10") for i in range(solve_length)))
        return result_str

    def run(self):
        for i in range(10000):
            print(i)
            self.population = sorted(self.population, key=lambda item: self.fitness_function(item))
            first_num, second_num = random.randint(0, self.population_length - 1), random.randint(0,
                                                                                                  self.population_length - 1)
            item1 = self.population[first_num]
            item2 = self.population[second_num]

            new1, new2 = self.crossover(item1, item2)
            self.population[0] = new1
            self.population[1] = new2

            for new in [new1, new2]:
                if self.solve_function(new):
                    return new

        return self.population[-1]

    @staticmethod
    def crossover(item1, item2):
        cross_point = random.randint(1, len(item1))
        return item1[0:cross_point] + item2[cross_point:len(item1)], item2[0:cross_point] + item1[
                                                                                            cross_point:len(item1)]


if __name__ == '__main__':
    def fitness(item):
        return sum([int(x) for x in item])


    def is_solve(item):
        for i in item:
            if i != "1":
                return False
        return True


    sga = SGA(population_length=100, solve_length=1000, fitness_function=fitness, solve_function=is_solve)
    print(sga.run())

