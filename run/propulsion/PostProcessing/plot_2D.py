import matplotlib.pyplot as plt
import niceplots
import pickle as pkl
import numpy as np
import os


def plot_sweeps():
    TOCw = np.zeros([10, 10])
    CRZw = np.zeros([10, 10])
    TSEC_TOC = np.zeros([10, 10])
    TSEC_CRZ = np.zeros([10, 10])

    for i in range(10):
        for j in range(10):
            with open(f"../OUTPUT/N3_trends/N3_sweeps/JetA/TOC-{i}_CRZ-{j}.pkl", "rb") as f:
                data = pkl.load(f)
                TOCw[i, j] = data[10]
                CRZw[i, j] = data[11]
                TSEC_TOC[i, j] = data[4]
                TSEC_CRZ[i, j] = data[7]

    # print(TOCw)
    # print(CRZw)
    fig = plt.figure(figsize=(12, 10))
    cp = plt.contourf(TOCw, CRZw, TSEC_CRZ)
    # plt.contourf(TOCw, CRZw, TSEC_TOC)
    fig.colorbar(cp)
    plt.title("Jet-A TSEC as a function of water recovery fraction")
    plt.xlabel("TOC water recovery fraction")
    plt.ylabel("CRZ water recovery fraction")
    # plt.show()
    fname = "CLVR_sweep"
    plt.savefig("plots/" + fname + ".pdf")
    plt.savefig("plots/" + fname + ".png")
    return


if __name__ == "__main__":
    niceplots.setRCParams()
    niceColors = niceplots.get_niceColors()
    plt.rcParams["font.size"] = 20

    plot_sweeps()

