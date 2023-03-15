import matplotlib.pyplot as plt
import niceplots
import pickle as pkl
import numpy as np
import os


def plot_NOx():
    plt.figure(figsize=(8, 6))
    T4 = [3400, 3600]

    colors = ["red", "green", "blue"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/EINOx_T4-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            xdata = data["humidity"]
            y1data = data["NOx_SLS"]
            y2data = data["NOx_CRZ"]

            plt.plot(xdata, y1data, color=colors[i], linestyle="dashed", label=f"SLS at T_4={T}R")
            plt.plot(xdata, y2data, color=colors[i], label=f"CRZ at T_4={T}R")

    plt.xlabel("Humidity Ratio of Atmosphere (kg/kg)")
    plt.ylabel("EINOx (g/kg)")
    # plt.ylabel("T_4 (R)")
    plt.legend()
    # plt.show()
    fname = "EINOx_hum1"
    fname = "EINOx_hum_T4"
    plt.savefig("plots/" + fname + ".pdf")


def plot_NOx_var(fx1, fx2, fy1, fy2, xname, yname, outname):
    plt.figure(figsize=(14, 10))
    T4 = [3400, 3500, 3600]

    colors = ["blue", "green", "red"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/EINOx_upd_T4-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            x1data = data[fx1]
            x2data = data[fx2]
            y1data = data[fy1]
            y2data = data[fy2]

            # plt.plot(x1data, y1data, color=colors[i], linestyle="dashed", label=f"SLS at RTO-T4={T}R")
            plt.plot(x2data, y2data, color=colors[i], label=r"CRZ at $T4_{RTO}$=" + f"{int(T)}R")

    plt.xlabel(xname)
    plt.ylabel(yname, rotation="horizontal", horizontalalignment="right")
    plt.legend(loc="best", fontsize=20)
    # plt.show()
    plt.savefig("plots/" + outname + ".png")


def plot_Hum_var(fx, fy1, fy2, xname, yname, outname):
    plt.figure(figsize=(8, 6))
    T4 = [3400, 3600]

    colors = ["red", "green", "blue"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/EINOx_T4-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            xdata = data[fx]
            y1data = data[fy1]
            y2data = data[fy2]

            plt.plot(xdata, y1data, color=colors[i], linestyle="dashed", label=f"SLS at RTO-T4={T}R")
            plt.plot(xdata, y2data, color=colors[i], label=f"CRZ at RTO-T4={T}R")

    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(loc="best")
    # plt.show()
    plt.savefig("plots/" + outname + ".pdf")


def plot_TSFC_inject():
    plt.figure(figsize=(14, 10))
    # T4 = [3300.0, 3400.0, 3500.0, 3600.0]
    T4 = [3400.0, 3500.0]

    colors = ["red", "green", "blue", "orange"]

    for i, T in enumerate(T4):
        fname = f"../OUTPUT/N3_trends/w_inject_H2-{T}.pkl"
        with open(fname, "rb") as f:
            data = pkl.load(f)

            xdata = data[0]
            ydata = data[1]

            plt.scatter(xdata[0], ydata[0], color=colors[i])
            plt.plot(xdata, ydata, color=colors[i], label=f"T_4={int(T)}R")
            # plt.plot([xdata[0], xdata[-1]], [ydata[0], ydata[0]], color=colors[i], linestyle="dashed", linewidth=1)

    plt.xlabel("Mass Flow Rate of Water Injected (lbm/s)")
    plt.ylabel("TSFC")
    plt.legend()
    plt.show()
    # fname = "TSFC_inject"
    # plt.savefig("plots/" + fname + ".pdf")


def plot_NOx_correlation():
    plt.figure(figsize=(8, 6))
    T3 = np.linspace(450, 850, 100)
    NOxP3 = 6.26e-8 * T3 ** 3 - (0.000117 * T3 ** 2) + (0.074 * T3) - 15.04

    plt.plot(T3, NOxP3)

    plt.xlabel("T3 (R)")
    plt.ylabel("EINOx/P3^0.4")
    plt.grid("on")

    plt.show()
    # fname = "TSFC_inject"
    # plt.savefig("plots/" + fname + ".pdf")


def plot_traj(TSFC, file1, file2, label1, label2):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    if TSFC:
        idx = 1
        ylabel = "TSFC"
    else:
        idx = 0
        ylabel = "EINOx (g/kg)"

    with open(file1, "rb") as f:
        data1 = pkl.load(f)

        y1 = np.array(data1[idx])

    with open(file2, "rb") as f:
        data2 = pkl.load(f)

        y2 = np.array(data2[idx])

    my_xticks = ["TOC", "RTO", "SLS", "CRZ"]
    x = [0, 1, 2, 3]

    # print(np.divide(y2 - y1, y1))
    # print(y1)
    # print(y2)

    ax[0].set_xticks(x, my_xticks)
    ax[0].plot(x, y1, clip_on=False, color=niceColors["Red"], label=label1)
    ax[0].plot(x, y2, clip_on=False, color=niceColors["Blue"], linestyle="--", label=label2)

    ax[0].set_ylabel(ylabel, rotation="horizontal", horizontalalignment="right")
    ax[0].legend()

    ax[1].set_xticks(x, my_xticks)
    ax[1].plot(x, np.divide(y2 - y1, y1) * 100, color=niceColors["Green"])

    ax[1].set_ylabel("% Diff. from " + label1, rotation="horizontal", horizontalalignment="right")
    # plt.show()
    fname = "TSFC_inject_" + label1 + "-" + label2
    plt.savefig("plots/" + fname + ".pdf")


def bar_traj(file1, file2, label1, label2):
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    with open(file1, "rb") as f:
        data1 = pkl.load(f)

        y1 = np.reshape(np.array(data1[1]), [4])

    with open(file2, "rb") as f:
        data2 = pkl.load(f)

        y2 = np.reshape(np.array(data2[1]), [4])

    # print(y1)
    # print(y2)
    header = ["Flight Condition", "TSFC, lbm/(h-lbf)"]
    labels = ["TOC", "RTO", "SLS", "CRZ"]
    names = [label1, label2]

    horiz_bar(labels, y1, y2, label1, label2, header, nd=4, size=[12, 0.5])
    niceplots.horiz_bar(labels, y1, header, nd=3)
    # for (y, ax) in zip(y2, axs):
    #     ax.scatter([y], [1], c=niceColors["Green"], lw=0, s=100, zorder=1, clip_on=False)

    # plt.savefig("plots/" + label1 + "_" + label2 + "_bar_chart.pdf", bbox_inches="tight")
    plt.savefig("plots/" + label1 + "_" + label2 + "_bar_chart_diff.png", dpi=400, bbox_inches="tight")


def horiz_bar(labels, times1, times2, name1, name2, header, nd=1, size=[5, 0.5], color=None):
    """Creates a horizontal bar chart to compare positive numbers.

    Parameters
    ----------
    labels : list of str
        contains the ordered labels for each data set
    times : list of float
        contains the numerical data for each entry
    header : list of two str
        contains the left and right header for the labels and
        numeric data, respectively
    nd : float
        the number of digits to show after the decimal point for the data
    size : list of two float
        the size of the final figure (iffy results)
    color : str
        hexcode for the color of the scatter points used

    Returns
    -------
    fig: matplotlib Figure
        Figure created
    ax: matplotlib Axes
        Axes on which data is plotted
    """

    # Use niceColours yellow if no colour specified
    niceColours = niceplots.get_niceColors()
    if color is None:
        color = niceColours["Yellow"]

    # Obtain parameters to size the chart correctly
    num = len(times1)
    width = size[0]
    height = size[1] * num
    t_max = max(times2)
    t_min = min(times2)

    # Create the corresponding number of subplots for each individual timing
    fig, axarr = plt.subplots(num, 1, figsize=[width, height])

    # Loop over each time and get the max number of digits
    t_max_digits = 0
    for t in times1:
        tm = len(str(int(t)))
        if tm > t_max_digits:
            t_max_digits = tm

    # Actual loop that draws each bar
    for j, (l, t1, t2, ax) in enumerate(zip(labels, times1, times2, axarr)):

        # Draw the gray line and singular yellow dot
        ax.axhline(y=1, c=niceColours["Grey"], lw=3, zorder=0, alpha=0.5)
        ax.scatter([t1], [1], c=niceColours["Green"], lw=0, s=100, zorder=1, clip_on=False)
        ax.scatter([t2], [1], c=niceColours["Yellow"], lw=0, s=100, zorder=1, clip_on=False)

        # Set chart properties
        ax.set_ylim(0.99, 1.01)
        ax.set_xlim(0, t_max * 1.05)  # min(t1, t2) max(t1, t2)
        ax.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            labelleft=False,
            labelright=False,
            labelbottom=False,
            right=False,  # ticks along the top edge are off
            bottom=j == num,
            top=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel(l, rotation="horizontal", ha="right", va="center")
        string1 = "{number:.{digits}f}".format(number=t1, digits=nd)
        ax.annotate(
            string1,
            xy=(1, 1),
            xytext=(6, 0),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            va="center",
        )
        string2 = "{number:.{digits}f}".format(number=t2, digits=nd)
        ax.annotate(
            string2,
            xy=(1, 1),
            xytext=(80, 0),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            va="center",
        )

        # Create the top bar line
        if j == 0:
            ax.annotate(
                name1,
                xy=(1, 1),
                xytext=(6, 15),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                va="center",
                color=niceColors["Green"],
            )
            ax.annotate(
                name2,
                xy=(1, 1),
                xytext=(80, 15),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                va="center",
                color=niceColors["Yellow"],
            )

            ax.text(t_min / 1.005, 1.02, header[0], ha="right", fontweight="bold", fontsize="large")  # min(t1, t2)
            ax.text(t_max * 1.005, 1.025, header[1], ha="left", fontweight="bold", fontsize="large")  # max(t1, t2)
            # ax.text(t_max + 0.05, 1.02, names[0], ha="left", fontsize=10, color=niceColors["Yellow"])
            # ax.text(t_max, 1.02, names[1], ha="left", fontsize=10, color=niceColors["Green"])

    return fig, axarr


def vert_bar(dirpath):
    niceColours = niceplots.get_niceColors()
    JetA_LHV = 18564  # BTU/lb
    LH2_LHV = 51591  # BTU/lb
    labels = ["TOC", "RTO", "SLS", "CRZ"]

    with open(dirpath + "N3_JetA_wet-air0.pkl", "rb") as f:
        data1 = pkl.load(f)
        TSFC_jd = np.reshape(np.array(data1[1]), [4])
        TSEC_jd = TSFC_jd * JetA_LHV

    with open(dirpath + "N3_JetA_wet-air05.pkl", "rb") as f:
        data1 = pkl.load(f)
        TSFC_jw = np.reshape(np.array(data1[1]), [4])
        TSEC_jw = TSFC_jw * JetA_LHV

    with open(dirpath + "N3_H2_wet-air0.pkl", "rb") as f:
        data1 = pkl.load(f)
        TSFC_hd = np.reshape(np.array(data1[1]), [4])
        TSEC_hd = TSFC_hd * LH2_LHV

    with open(dirpath + "N3_H2_wet-air009.pkl", "rb") as f:
        data1 = pkl.load(f)
        TSFC_hw = np.reshape(np.array(data1[1]), [4])
        TSEC_hw = TSFC_hw * LH2_LHV

    rel_TSEC_jw = -(TSEC_jw - TSEC_jd) / TSEC_jd * 100
    rel_TSEC_hw = -(TSEC_hw - TSEC_hd) / TSEC_hd * 100
    print(TSEC_jd)
    print(TSEC_hd)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=[12, 8])
    # fig, ax = plt.subplots()
    width = 0.5

    ax[0].bar(labels, rel_TSEC_jw, width, label="Jet-A with Water Injection", color=niceColours["Yellow"])
    ax[1].bar(labels, rel_TSEC_hw, width, label="LH2 with Water Injection", color=niceColours["Blue"])
    # ax.bar(labels, rel_TSEC_jw, width, label="Jet-A")
    # ax.bar(labels, rel_TSEC_hw, width, bottom=rel_TSEC_jw, label="LH2")

    ax[0].set_ylabel("Percent Relative\nImprovement in TSEC", rotation="horizontal", ha="right", va="center")
    ax[1].set_ylabel("Percent Relative\nImprovement in TSEC", rotation="horizontal", ha="right", va="center")
    # ax.set_ylabel("Relative Difference in TSEC with Water Injection")
    # ax.set_title("Relative Improvements of Water Injection")
    ax[0].legend(loc="upper center")
    ax[1].legend(loc="upper center")
    # ax.legend()
    plt.savefig("plots/JetA-H2_bar_chart_diff.pdf", dpi=400, bbox_inches="tight")
    plt.savefig("plots/JetA-H2_bar_chart_diff.png", dpi=400, bbox_inches="tight")

    # plt.show()

    return


def vert_bar_CLVR(dirpath):
    niceColours = niceplots.get_niceColors()
    labels = ["TOC", "RTO", "SLS", "CRZ"]

    TSEC_jw = np.array([8091.83600, 5158.95131, 3108.50255, 7886.58356])
    TSEC_jd = np.array([8159.47841, 5106.91800, 3094.18795, 8199.00998])
    TSEC_hw = np.array([8161.40341, 5415.06315, 3250.09278, 7840.04626])
    TSEC_hd = np.array([8523.68744, 5284.35445, 3202.25020, 8518.99365])

    rel_TSEC_jw = -(TSEC_jw - TSEC_jd) / TSEC_jd * 100
    rel_TSEC_hw = -(TSEC_hw - TSEC_hd) / TSEC_hd * 100
    # print(TSEC_jd)
    # print(TSEC_hd)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=[12, 8])
    # fig, ax = plt.subplots()
    width = 0.5

    ax[0].bar(labels, rel_TSEC_jw, width, label="Jet-A with water recovery", color=niceColours["Yellow"])
    ax[1].bar(labels, rel_TSEC_hw, width, label=r"$LH_2$ with water recovery", color=niceColours["Blue"])
    # ax.bar(labels, rel_TSEC_jw, width, label="Jet-A")
    # ax.bar(labels, rel_TSEC_hw, width, bottom=rel_TSEC_jw, label="LH2")

    ax[0].set_ylabel("Percent Relative\nImprovement in TSEC", rotation="horizontal", ha="right", va="center")
    ax[1].set_ylabel("Percent Relative\nImprovement in TSEC", rotation="horizontal", ha="right", va="center")
    # ax.set_ylabel("Relative Difference in TSEC with Water Injection")
    # ax.set_title("Relative Improvements of Water Injection")
    ax[0].legend(loc="upper center")
    ax[1].legend(loc="upper center")
    # ax.legend()
    if os.path.isdir(dirpath) is False:
        os.mkdir(dirpath)
    plt.savefig(dirpath + "JetA-H2_TSEC_diff.pdf", dpi=400, bbox_inches="tight")
    plt.savefig(dirpath + "JetA-H2_TSEC_diff.png", dpi=400, bbox_inches="tight")

    # plt.show()

    return


def plot_TSFC_wfrac_comb(fname):
    plt.figure(figsize=(14, 10))

    with open(fname, "rb") as f:
        data = pkl.load(f)

        with open("../OUTPUT/N3_trends/N3_wfrac_H2_0-08_TOC.pkl", "rb") as f:
            data2 = pkl.load(f)

            xdata = np.append(data[0], data2[0])
            # xdata = data[1]
            y1data = np.append(data[2], data2[2])
            y2data = np.append(data[3], data2[3])

    # plt.plot(xdata, (y1data - y1data[-1]) / y1data[-1] * 100, label="CRZ", linewidth=5)
    # plt.plot(xdata, (y2data - y2data[-1]) / y2data[-1] * 100, label="TOC", linewidth=5)
    plt.plot(xdata, (y1data - y1data[0]) / y1data[0] * 100, label="CRZ (0% water recovered)", linewidth=5)
    plt.plot(xdata, (y2data - y2data[0]) / y2data[0] * 100, label="TOC", linewidth=5)
    # plt.plot(xdata, (y1data - y1data[-1]) / y1data[-1] * 100, label="CRZ", linewidth=5)
    # plt.plot(
    #     xdata,
    #     (y2data - y2data[-1]) / y2data[-1] * 100,
    #     label="TOC, (constant 10% exhaust water recovery)",
    #     linewidth=5,
    # )
    # plt.scatter([0.06148998], [0.43719117], color="red", s=100, zorder=1, label="SNOPT Optimum")

    plt.xlabel("Fraction of core exhaust water recovered (TOC)")
    # plt.xlabel("Fraction of core exhaust water recovered (CRZ)")
    # plt.xlabel("Water flow rate (lbm/s)")
    plt.ylabel("Percent\nDifference TSFC", rotation="horizontal", ha="right", va="center")
    plt.legend()
    # plt.show()
    fname = "TSFC_N3-CLVR-H2-05-TOC"
    # fname = "TSFC_N3-CLVR-29-CRZ"
    # fname = "TSFC_N3-inject-upd"
    plt.savefig("plots/" + fname + ".pdf")
    plt.savefig("plots/" + fname + ".png")


def plot_NOx_wfrac(fname):
    plt.figure(figsize=(18, 6))

    with open(fname, "rb") as f:
        data = pkl.load(f)

        xdata = data[0]
        # xdata = data[1]
        TOCdata = data[10]
        RTOdata = data[11]
        SLSdata = data[12]
        CRZdata = data[13]

    plt.plot(xdata, (TOCdata - TOCdata[0]) / TOCdata[0] * 100, label="TOC (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (TOCdata), label="TOC", linewidth=5)
    plt.plot(xdata, (RTOdata - RTOdata[0]) / RTOdata[0] * 100, label="RTO (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (RTOdata - RTOdata[0]) / RTOdata[0] * 100, label="RTO", linewidth=5)
    plt.plot(xdata, (SLSdata - SLSdata[0]) / SLSdata[0] * 100, label="SLS (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (SLSdata - SLSdata[0]) / SLSdata[0] * 100, label="SLS", linewidth=5)
    # plt.plot(xdata, (CRZdata), label="CRZ (0% water recovered)", linewidth=5)
    plt.plot(xdata, (CRZdata - CRZdata[0]) / CRZdata[0] * 100, label="CRZ", linewidth=5)

    # plt.xlabel("Fraction of core exhaust water recovered (TOC)")
    # plt.xlabel("Fraction of core exhaust water recovered (RTO)")
    # plt.xlabel("Fraction of core exhaust water recovered (SLS)")
    plt.xlabel("Fraction of core exhaust water recovered (CRZ)")
    # plt.xlabel("Water flow rate (lbm/s)")
    plt.ylabel("EINOx (g/kg)", rotation="horizontal", ha="right", va="center")
    plt.legend()
    # plt.show()
    # fname = "TSEC_N3-CLVR-H2-0-10-TOC"
    # fname = "TSEC_N3-CLVR-H2-0-5-RTO"
    # fname = "TSEC_N3-CLVR-H2-0-5-SLS"
    fname = "TSEC_N3-CLVR-JetA-0-10-CRZ"
    outdir = "plots/NOx/"
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(outdir + fname + ".pdf")
    plt.savefig(outdir + fname + ".png")


def plot_TSEC_wfrac(fname):
    plt.figure(figsize=(18, 6))

    with open(fname, "rb") as f:
        data = pkl.load(f)

        xdata = data[0]
        # xdata = data[1]
        TOCdata = data[6]
        RTOdata = data[7]
        SLSdata = data[8]
        CRZdata = data[9]

    # plt.plot(xdata, (TOCdata - TOCdata[0]) / TOCdata[0] * 100, label="TOC (0% water recovered)", linewidth=5)
    plt.plot(xdata, (TOCdata - TOCdata[0]) / TOCdata[0] * 100, label="TOC", linewidth=5)
    plt.plot(xdata, (RTOdata - RTOdata[0]) / RTOdata[0] * 100, label="RTO (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (RTOdata - RTOdata[0]) / RTOdata[0] * 100, label="RTO", linewidth=5)
    plt.plot(xdata, (SLSdata - SLSdata[0]) / SLSdata[0] * 100, label="SLS (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (SLSdata - SLSdata[0]) / SLSdata[0] * 100, label="SLS", linewidth=5)
    plt.plot(xdata, (CRZdata - CRZdata[0]) / CRZdata[0] * 100, label="CRZ (0% water recovered)", linewidth=5)
    # plt.plot(xdata, (CRZdata - CRZdata[0]) / CRZdata[0] * 100, label="CRZ", linewidth=5)

    plt.xlabel("Fraction of core exhaust water recovered (TOC)")
    # plt.xlabel("Fraction of core exhaust water recovered (RTO)")
    # plt.xlabel("Fraction of core exhaust water recovered (SLS)")
    # plt.xlabel("Fraction of core exhaust water recovered (CRZ)")
    # plt.xlabel("Water flow rate (lbm/s)")
    plt.ylabel("Percent\nDifference TSEC", rotation="horizontal", ha="right", va="center")
    plt.legend()
    # plt.show()
    fname = "TSEC_N3-CLVR-H2-0-7-TOC"
    # fname = "TSEC_N3-CLVR-H2-0-5-RTO"
    # fname = "TSEC_N3-CLVR-H2-0-5-SLS"
    # fname = "TSEC_N3-CLVR-H2-0-10-CRZ"
    outdir = "plots/TSEC/"
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(outdir + fname + ".pdf")
    plt.savefig(outdir + fname + ".png")


def plot_TSFC_compare(fname1, fname2):
    plt.figure(figsize=(14, 10))

    with open(fname1, "rb") as f:
        data = pkl.load(f)

        # xdata = data[0]
        xdata = data[1]
        y1data = data[2]
        y2data = data[3]

        # plt.plot(xdata, y1data, label="CRZ")
        plt.plot(xdata, y2data, label="Water Recovery")

    with open(fname2, "rb") as f:
        data = pkl.load(f)

        xdata = data[0]
        y1data = data[1]
        y2data = data[2]

        # plt.plot(xdata, y1data, label="CRZ")
        plt.plot(xdata, y2data, label="Water Injection")
        plt.plot(xdata, y2data[-1] * np.ones(y2data.size), label="No Injection")

    # plt.xlabel("Fraction of water recovered")
    plt.xlabel("Water flow rate (lbm/s)")
    plt.ylabel("TSFC (TOC)")
    plt.legend()
    # plt.show()
    fname = "TSFC_N3-compare"
    plt.savefig("plots/" + fname + ".pdf")
    plt.savefig("plots/" + fname + ".png")


if __name__ == "__main__":
    niceplots.setRCParams()
    niceColors = niceplots.get_niceColors()
    plt.rcParams["font.size"] = 20

    # plot_TSEC_wfrac("../OUTPUT/N3_trends/N3_wfrac_H2_0-7_TOC.pkl")
    # plot_TSEC_wfrac("../OUTPUT/N3_trends/N3_wfrac_H2_0-5_RTO.pkl")
    # plot_TSEC_wfrac("../OUTPUT/N3_trends/N3_wfrac_H2_0-5_SLS.pkl")
    # plot_TSEC_wfrac("../OUTPUT/N3_trends/N3_wfrac_H2_0-10_CRZ.pkl")
    # plot_NOx_wfrac("../OUTPUT/N3_trends/N3_wfrac_JetA_0-10_TOC.pkl")
    # plot_NOx_wfrac("../OUTPUT/N3_trends/N3_wfrac_JetA_0-10_CRZ.pkl")
    # plot_TSFC_wfrac("../OUTPUT/N3_trends/N3_wfrac_JetA_10_TOC.pkl")
    # plot_TSFC_wfrac("../OUTPUT/N3_trends/N3_wfrac_JetA_29_CRZ.pkl")
    # plot_TSFC_compare("../OUTPUT/N3_trends/N3_wfrac_JetA_08_TOC.pkl", "../OUTPUT/N3_trends/w_inject_JetA.pkl")
    # plot_TSFC_wfrac("../OUTPUT/N3_trends/w_inject_JetA.pkl")
    # plot_TSFC_wfrac("../OUTPUT/N3_trends/w_inject_JetA-3400.0_wAREA_HPC53.pkl")

    vert_bar_CLVR("plots/bar_plot/")
    # vert_bar("../OUTPUT/N3_trends/")

    # bar_traj(
    #     file2="../OUTPUT/N3_trends/N3_JetA_wet-air0.pkl",
    #     file1="../OUTPUT/N3_trends/N3_JetA_wet-air05.pkl",
    #     label2="JetA, Dry",
    #     label1="JetA, Wet",
    # )

    # bar_traj(
    #     file2="../OUTPUT/N3_trends/N3_H2_wet-air0.pkl",
    #     file1="../OUTPUT/N3_trends/N3_H2_wet-air009.pkl",
    #     label2="H2, Dry",
    #     label1="H2, Wet",
    # )

    # bar_traj(
    #     file1="../OUTPUT/N3_trends/N3_H2_wet-air0.pkl",
    #     file2="../OUTPUT/N3_trends/N3_JetA_wet-air0.pkl",
    #     label2="JetA",
    #     label1="H2",
    # )

    # plot_traj(
    #     TSFC=True,
    #     file1="../OUTPUT/N3_trends/N3_JetA_wet-air0.pkl",
    #     file2="../OUTPUT/N3_trends/N3_JetA_wet-air05.pkl",
    #     label1="JetA, Dry",
    #     label2="JetA, Wet",
    # )

    # plot_traj(
    #     TSFC=True,
    #     file1="../OUTPUT/N3_trends/N3_H2_wet-air0.pkl",
    #     file2="../OUTPUT/N3_trends/N3_H2_wet-air009.pkl",
    #     label1="H2, Dry",
    #     label2="H2, Wet",
    # )

    # plot_traj(
    #     TSFC=True,
    #     file2="../OUTPUT/N3_trends/N3_H2_wet-air0.pkl",
    #     file1="../OUTPUT/N3_trends/N3_JetA_wet-air0.pkl",
    #     label1="JetA",
    #     label2="H2",
    # )

    # plot_NOx_correlation()
    # plot_NOx()
    # plot_NOx_var(
    #     fx1="T4_SLS",
    #     fx2="T4_CRZ",
    #     fy1="NOx_SLS",
    #     fy2="NOx_CRZ",
    #     xname="T4 (R)",
    #     yname="EINOx (g/kg)",
    #     outname="EINOx_T4",
    # )
    # plot_NOx_var(
    #     fx1="T3_SLS",
    #     fx2="T3_CRZ",
    #     fy1="NOx_SLS",
    #     fy2="NOx_CRZ",
    #     xname="T3 (R)",
    #     yname="EINOx (g/kg)",
    #     outname="EINOx_T3",
    # )
    # plot_NOx_var(
    #     fx1="P3_SLS",
    #     fx2="P3_CRZ",
    #     fy1="NOx_SLS",
    #     fy2="NOx_CRZ",
    #     xname="P3 (psi)",
    #     yname="EINOx (g/kg)",
    #     outname="EINOx_P3",
    # )

    # plot_NOx_var(
    #     fy1="T4_SLS",
    #     fy2="T4_CRZ",
    #     fx1="humidity",
    #     fx2="humidity",
    #     xname="Humidity ratio (kg/kg)",
    #     yname=r"$T4_{CRZ}$ (R)",
    #     outname="Hum_T4",
    # )

    # plot_Hum_var(
    #     fx="humidity",
    #     fy1="T4_SLS",
    #     fy2="T4_CRZ",
    #     xname="Humidity Ratio of Atmosphere (kg/kg)",
    #     yname="Off-Design T4 (R)",
    #     fname="Hum_T4",
    # )
    # plot_Hum_var(
    #     fx="humidity",
    #     fy1="T3_SLS",
    #     fy2="T3_CRZ",
    #     xname="Humidity Ratio of Atmosphere (kg/kg)",
    #     yname="Off-Design T3 (R)",
    #     fname="Hum_T3",
    # )
    # plot_Hum_var(
    #     fx="humidity",
    #     fy1="P3_SLS",
    #     fy2="P3_CRZ",
    #     xname="Humidity Ratio of Atmosphere (kg/kg)",
    #     yname="Off-Design P3 (psi)",
    #     fname="Hum_P3",
    # )

    # plot_TSFC_inject()
