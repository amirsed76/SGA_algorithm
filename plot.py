import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sql_manager import SqlManager
from SGA import one_max, peak, trap
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


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


def calculate_color(row):
    problem_size = int(row["problem_size"])
    if problem_size == 10:
        row["color"] = "y"
    elif problem_size == 30:
        row["color"] = "g"
    elif problem_size == 50:
        row["color"] = "b"
    elif problem_size == 70:
        row["color"] = "m"
    elif problem_size == 100:
        row["color"] = "r"

    return row


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql(sql="select * from information", con=sql_manager.conn)
    for func in [one_max, peak, trap]:
        func_df = query_by_fitness_function(func.__name__, df)
        func_df = func_df.apply(calculate_fitness_value_for_each_row, axis=1, args=[func])
        print("____________________")
        # print(func_df)
        group_df = func_df.groupby(["problem_size", "pop_size", "generation"]).agg(
            {'fitness_value': ['mean', 'std']}).reset_index()

        X = group_df["pop_size"]
        Y = group_df["generation"]
        Z = group_df["fitness_value"]["mean"]

        fig = plt.figure(func.__name__)
        ax = fig.add_subplot(111, projection='3d')
        group_df = group_df.apply(calculate_color, axis=1)
        ax.scatter(X, Y, Z, c=group_df["color"], marker='o')

        ax.set_xlabel("pop_size")
        ax.set_ylabel("generation")
        ax.set_zlabel('fitness_value')

        x = np.linspace(0, 1, 10)
        for i, color in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')], start=1):
            plt.plot(x, i * x + i, color=color[1], label=str(color[0]))
        plt.legend(loc='best')
        plt.show()
