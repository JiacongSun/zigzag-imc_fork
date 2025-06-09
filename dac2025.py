import pickle
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np


def read_pickle(filename="result.pkl"):
    with open(filename, "rb") as fp:
        df = pickle.load(fp)
    return df

def normalize_vector(vec):
    # @param vec: input numpy vector
    min_value = np.min(vec)
    assert min_value > 0
    norm_vec = vec / min_value
    return norm_vec

def metric_comparison(df, workload="resnet8"):
    ##############################
    ## setting
    acc_type = "DIMC"
    ci_op = 300 / 3.6e+18  # g,CO2/pJ
    ci_em = 9.17  # g,CO2/mm2@28nm
    chip_yield = 0.95
    lifetime = 94608000e+9  # 3 years in ns
    ##############################
    df_cc = df[(df.acc_type == acc_type) & (df.workload == workload) & (df.sram_size >= 131072) & (df.dim >= 256)]
    # df_cc = df[(df.acc_type == acc_type) & (df.workload == workload)]
    # df_cc = df

    delay = df_cc["t_lat"] * df_cc["t_tclk"]  # ns
    area = df_cc["t_area"]  # mm2
    energy = df_cc["t_en"]  # pJ
    # TODO: calc RCI
    rci = ci_op * lifetime / (ci_em / chip_yield)
    # TODO: calc Cop
    c_op = energy * ci_op  # g,CO2
    # TODO: calc Cem
    c_em = delay * area * ci_em / (chip_yield * lifetime)  # g,CO2
    # TODO: calc Ctot
    c_tot = c_op + c_em  # g,CO2
    # TODO: calc metric 1: EDP
    metric_edp = energy * delay  # pJ * ns
    metric_edp = normalize_vector(metric_edp)
    # TODO: calc metric 2: CDP
    metric_cdp = c_tot * delay  # g,CO2 * ns
    metric_cdp = normalize_vector(metric_cdp)
    # TODO: calc metric 3: CEP
    metric_cep = c_tot * energy  # g,CO2 * pJ
    metric_cep = normalize_vector(metric_cep)
    # TODO: calc metric 4: C2EP
    metric_c2ep = c_tot * c_tot * energy  # g,CO2^2 * pJ
    metric_c2ep = normalize_vector(metric_c2ep)
    # TODO: calc metric 5: CE2P
    metric_ce2p = c_tot * energy * energy  # g,CO2 * pJ^2
    metric_ce2p = normalize_vector(metric_ce2p)
    # TODO: calc metric 6: tCDP
    metric_tcdp = c_em * delay  # g,CO2 * ns
    metric_tcdp = normalize_vector(metric_tcdp)
    # TODO: calc metric 7: CNADP
    metric_cnadp = area * delay + (energy * rci)
    metric_cnadp = normalize_vector(metric_cnadp)
    # TODO: calc metric 8: normalized c_tot
    c_tot = normalize_vector(c_tot)
    # TODO: plot
    index = np.arange(len(metric_cnadp))
    metrics = [metric_edp, metric_cdp, metric_cep, metric_c2ep, metric_ce2p, metric_tcdp, metric_cnadp, c_tot]
    labels = ["EDP", "CDP", "CEP", "C2EP", "CE2P", "tCDP", "CNADP", "C"]
    # metrics = [metric_cep, metric_cnadp, c_tot]
    # labels = ["CEP", "CNADP", "C"]
    for i, vec in enumerate(metrics):
        logging.info(f"Index {i} of min: {np.argmin(vec)}")
    width = 0.1
    for i, vec in enumerate(metrics):
        # plt.bar(index + i * width, vec, color="skyblue", edgecolor="black", width=0.1)
        if i == len(metrics) - 1:
            plt.plot(index, vec, label=f"{labels[i]}", linestyle=":", color="black", linewidth=2)
        else:
            plt.plot(index, vec, label=f"{labels[i]}")
    plt.xlabel("HW cases")
    plt.ylabel("Normalized metrics")
    plt.grid(visible=True, which="both", axis="both")
    plt.legend()
    # plt.ylim([0, 5])
    plt.tight_layout()
    plt.show()
    pass

def plot_carbon_breakdown(df, workload="resnet8"):
    ##############################
    ## setting
    acc_type = "DIMC"
    ci_op = 41/3.6e+18  # g,CO2/pJ
    ci_em = 9.17  # g,CO2/mm2@28nm
    chip_yield = 0.95
    lifetime = 94608000e+9  # 3 years in ns
    ##############################
    df_cc = df[(df.acc_type == acc_type) & (df.workload == workload) & (df.sram_size >= 131072) & (df.dim >= 256)]
    delay = df_cc["t_lat"] * df_cc["t_tclk"]  # ns
    area = df_cc["t_area"]  # mm2
    energy = df_cc["t_en"]  # pJ
    # TODO: calc Cop
    c_op = energy * ci_op
    # TODO: calc Cem
    c_em = delay * area * ci_em / (chip_yield * lifetime)
    # TODO: plot
    index = [i for i in range(len(c_op))]
    plt.bar(index, c_op, color="skyblue", edgecolor="black")
    plt.bar(index, c_em, bottom=c_op, color="orange", edgecolor="black")
    plt.xlabel("HW cases")
    plt.ylabel("Ctot [g, CO2]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging_level = logging.INFO  # logging level
    logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging_level, format=logging_format)
    pd.set_option('display.max_colwidth', None)
    #####################################
    ## experiment setting
    workload_suit = "tiny"
    #####################################

    if workload_suit == "tiny":
        df = read_pickle("no_cme_expr_res_tiny.pkl")
        workloads = ["ae", "ds_cnn", "mobilenet", "resnet8"]  # legal workload keywords
        sram_sizes = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024]  # unit: B
        workloads.append("geo")  # append geo so that plotting for geo is also supported
    elif workload_suit == "mobile":
        df = read_pickle("no_cme_expr_res_mobile.pkl")
        workloads = ["deeplabv3", "mobilebert", "mobilenet_edgetpu", "mobilenet_v2"]  # legal workload keywords
        sram_sizes = [1 * 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]  # unit: B
        workloads.append("geo")  # append geo so that plotting for geo is also supported
    else:  # debugging branch
        pass
    """plot"""
    for wk in ["resnet8"]:
        # plot_carbon_breakdown(df=df, workload=wk)
        metric_comparison(df=df, workload=wk)
    breakpoint()