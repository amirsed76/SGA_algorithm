import pandas as pd
from sql_manager import SqlManager
from SGA import one_max, peak, trap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def str_to_list(string):
    return [int(ch) for ch in string]


def query_by_fitness_function(function_name, data_frame):
    return data_frame.query(f"fitness == '{function_name}'")


def query_by_problem_size(problem_size, data_frame):
    return data_frame.query(f"problem_size == {problem_size}")


def query_by_population_size(pop_size, data_frame):
    return data_frame.query(f"pop_size == {pop_size}")


def calculate_fitness_value(string, fitness_function):
    return fitness_function(str_to_list(string))


def calculate_fitness_value_for_each_row(row, fitness_function):
    row["fitness_value"] = calculate_fitness_value(row["result"], fitness_function)
    return row


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql(sql="select * from information", con=sql_manager.conn)
    for func in [one_max, peak, trap]:
        func_df = query_by_fitness_function(func.__name__, df)
        func_df = func_df.apply(calculate_fitness_value_for_each_row, axis=1, args=[func])
        print("____________________")
        # print(func_df)
        group_df = func_df.groupby(["problem_size", "pop_size", "generation"]).agg({'fitness_value': ['mean', 'std']})
        # for problem_size in [10, 30, 50, 70, 100]:
        #     # select color
        #     problem_size_df = query_by_problem_size(problem_size, func_df)
        #     for pop_size in [50, 100, 200, 300]:
        #         pop_size_df = query_by_population_size(pop_size, problem_size_df)
        #         pop_size_df = pop_size_df.apply(calculate_fitness_value_for_each_row, axis=1, args=[func])
        #         # print(pop_size_df)
        #         # print("________________")
        #         # print("fitness_function : ", func.__name__)
        #         # print("problem_size : ", problem_size)
        #         # print("population_size : ", pop_size)
        #         # group_df = pop_size_df.groupby(["generation"]).agg({'fitness_value': ['mean', 'std']})
        #         # group_df[]
        #         # fig = plt.figure()
        #         # ax = Axes3D(fig)
        #         # ax.plot_surface(X=)
