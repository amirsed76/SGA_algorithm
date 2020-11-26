import pandas as pd

from sql_manager import SqlManager
from SGA import one_max, peak, trap
import numpy as np
import matplotlib.pyplot as plt


def str_to_list(string):
    return [int(ch) for ch in string]


def query_by_fitness_function(function_name, data_frame):
    return data_frame.query(f"fitness == '{function_name}'")


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


def draw_plot(data_frame2, name):
    for state in ["mean", "std"]:
        fig = plt.figure(name + "  " + state)
        ax = fig.add_subplot(111, projection='3d')
        for color in ['y', 'g', 'b', 'm', 'r']:
            data_frame = data_frame2.loc[data_frame2["color"] == color]
            X = data_frame["pop_size"]
            Y = data_frame["max_gen"]
            Z = data_frame["fitness_value"][state]
            colors = data_frame["color"]
            ax.scatter(X, Y, Z, c=colors, marker='o')
            ax.plot(X, Y, Z, color=color)

        ax.set_xlabel("pop_size")
        ax.set_ylabel("max_gen")
        ax.set_zlabel(f'fitness_value({state})')

        x = np.linspace(0, 1, 10)
        for i, color in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')], start=1):
            plt.plot(x, i * x + i, color=color[1], label=str(color[0]))
        plt.legend(loc='best')

        plt.show()


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql(sql="select * from information", con=sql_manager.conn)
    for func in [one_max, peak, trap]:
        func_df = query_by_fitness_function(func.__name__, df)
        func_df = func_df.apply(calculate_fitness_value_for_each_row, axis=1, args=[func])
        print("____________________")
        # print(func_df)
        group_df = func_df.groupby(["problem_size", "pop_size", "max_gen"]).agg(
            {'fitness_value': ['mean', 'std']}).reset_index()
        group_df = group_df.apply(calculate_color, axis=1)
        group_df: pd.DataFrame
        # print(group_df["color"])
        draw_plot(data_frame2=group_df, name=func.__name__)
