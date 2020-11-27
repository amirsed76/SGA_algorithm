from pprint import pprint

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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


def draw_plot(input_df, name):
    for state in ["mean", "std"]:
        fig = plt.figure(name + "  " + state)
        ax = fig.add_subplot(111, projection='3d')
        for color in ['y', 'g', 'b', 'm', 'r']:
            data_frame = input_df.loc[input_df["color"] == color]
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


def draw_plot2(input_df, name):
    fig = plt.figure(f"{name} ")

    ax = Axes3D(fig)

    for index, (prob_size, color) in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')]):
        data_frame = input_df.loc[input_df["problem_size"] == prob_size]
        pop_sizes = input_df["pop_size"].copy().drop_duplicates()
        max_gens = input_df["max_gen"].copy().drop_duplicates()
        df2 = data_frame.copy()
        data = []
        errors = []
        for max_gen in max_gens.tolist():
            raw = []
            error_raw = []
            max_gen_df = df2.loc[df2["max_gen"] == max_gen].copy()
            for pop_size in pop_sizes.tolist():
                value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["fitness_value"]["mean"].values[0]
                error_value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["fitness_value"]["std"].values[0]
                raw.append(value)
                error_raw.append(error_value)
            data.append(raw)
            errors.append(error_raw)

        data = np.array(data)
        error_data = np.array(errors)

        lx = len(data[0])  # Work out matrix dimensions
        ly = len(data[:, 0])
        xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + index / 10, ypos + 0.05)
        #
        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        dx = 0.1 * np.ones_like(zpos)
        dy = dx.copy()
        dz = data.flatten()
        d_err = error_data.flatten()
        err_pos = dz.copy() - d_err / 2

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
        ax.bar3d(xpos, ypos, err_pos, 0.1, 0.1, d_err, color="black")
    r = np.linspace(0, 1, 10)
    for i, item in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')], start=1):
        plt.plot(0, 0, color=item[1], label=str(item[0]))
    plt.legend(loc='best')
    ax.set_xlabel("POP_SIZE")
    ax.set_ylabel('MAX_GEN')
    ax.set_zlabel('FITNESS')
    x_labels = []
    for x in pop_sizes.tolist():
        x_labels.append(" ")
        x_labels.append(x)

    y_labels = []
    for y in max_gens.tolist():
        y_labels.append(" ")
        y_labels.append(y)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()


def draw_plot3(input_df, name):
    fig = plt.figure(f"{name} ")

    ax = Axes3D(fig)

    for index, (prob_size, color) in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')]):
        data_frame = input_df.loc[input_df["problem_size"] == prob_size]
        pop_sizes = input_df["pop_size"].copy().drop_duplicates()
        max_gens = input_df["max_gen"].copy().drop_duplicates()
        df2 = data_frame.copy()
        data = []
        errors = []
        for max_gen in max_gens.tolist():
            raw = []
            error_raw = []
            max_gen_df = df2.loc[df2["max_gen"] == max_gen].copy()
            for pop_size in pop_sizes.tolist():
                value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["time"]["mean"].values[0]
                error_value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["time"]["std"].values[0]
                raw.append(value)
                error_raw.append(error_value)
            data.append(raw)
            errors.append(error_raw)

        data = np.array(data)
        error_data = np.array(errors)

        lx = len(data[0])  # Work out matrix dimensions
        ly = len(data[:, 0])
        xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + index / 10, ypos + 0.05)
        #
        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        dx = 0.1 * np.ones_like(zpos)
        dy = dx.copy()
        dz = data.flatten()
        d_err = error_data.flatten()
        err_pos = dz.copy() - d_err / 2

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
        ax.bar3d(xpos, ypos, err_pos, 0.1, 0.1, d_err, color="black")
    r = np.linspace(0, 1, 10)
    for i, item in enumerate([(10, 'y'), (30, 'g'), (50, 'b'), (70, 'm'), (100, 'r')], start=1):
        plt.plot(0, 0, color=item[1], label=str(item[0]))
    plt.legend(loc='best')
    ax.set_xlabel("POP_SIZE")
    ax.set_ylabel('MAX_GEN')
    ax.set_zlabel('TIME')
    x_labels = []
    for x in pop_sizes.tolist():
        x_labels.append(" ")
        x_labels.append(x)

    y_labels = []
    for y in max_gens.tolist():
        y_labels.append(" ")
        y_labels.append(y)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql(sql="select * from information", con=sql_manager.conn)
    for func in [one_max, peak, trap]:
        func_df = query_by_fitness_function(func.__name__, df)
        func_df = func_df.apply(calculate_fitness_value_for_each_row, axis=1, args=[func])
        func_df["time"] = func_df["generation"] * func_df["pop_size"]
        print("____________________")
        group_df = func_df.groupby(["problem_size", "pop_size", "max_gen"]).agg(
            {'fitness_value': ['mean', 'std'], 'time': ['mean', 'std']}).reset_index()
        group_df = group_df.apply(calculate_color, axis=1)
        draw_plot(input_df=group_df, name=func.__name__)
        draw_plot2(group_df, func.__name__)
        draw_plot3(group_df, func.__name__)
