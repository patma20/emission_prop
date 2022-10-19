import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def plot_NOx():
    plt.figure(figsize=(8, 6))
    T4 = [3200.0, 3400.0, 3600.0]

    colors = ["red", "green", "blue"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/EINOx_T4-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            xdata = data[0]
            y1data = data[1]
            y2data = data[2]

            plt.plot(xdata, y1data, color=colors[i], linestyle="dashed", label=f"SLS at T_4={T}R")
            plt.plot(xdata, y2data, color=colors[i], label=f"CRZ at T_4={T}R")

    plt.xlabel("Humidity Ratio of Atmosphere (kg/kg)")
    plt.ylabel("EINOx (g/kg)")
    plt.legend()
    # plt.show()
    fname = "EINOx_hum"
    plt.savefig(fname + ".pdf")


def plot_TSFC_inject():
    plt.figure(figsize=(8, 6))
    T4 = [3300.0, 3400.0, 3500.0, 3600.0]

    colors = ["red", "green", "blue", "orange"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/w_inject-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            xdata = data[0]
            ydata = data[1]

            plt.scatter(xdata[0], ydata[0], color=colors[i])
            plt.plot(xdata, ydata, color=colors[i], label=f"T_4={int(T)}R")
            plt.plot([xdata[0], xdata[-1]], [ydata[0], ydata[0]], color=colors[i], linestyle="dashed", linewidth=1)

    plt.xlabel("Mass Flow Rate of Water Injected (lbm/s)")
    plt.ylabel("TSFC")
    plt.legend()
    # plt.show()
    fname = "TSFC_inject"
    plt.savefig(fname + ".pdf")


if __name__ == "__main__":
    plot_NOx()
    plot_TSFC_inject()
