from zigzag.api import get_hardware_performance_zigzag
import math, re
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import os
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.inputs.examples.hardware.Aimc import memory_hierarchy_dut
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.accelerator import Accelerator
import json


def plot_curve(i_df, imc_types, workload):
    colors = [u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    markers = ["o", "^"]
    width = 0.35
    font_size = 15
    df = i_df[i_df.workload == workload]
    df["imc"] = df.dim.astype(str) + f"$\\times$" + df.dim.astype(str)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    topsw_cur = axs[0, 0]
    tops_cur = axs[0, 1] 
    topsmm2_cur = axs[1, 0]
    cf_cur = axs[1, 1]  # CF/inference
    for ii_a, a in enumerate(imc_types):
        dff = df[df.imc_type == a]
        labels = dff.imc.to_numpy()  # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))

        # Plot the curve
        # Plot topsw
        topsw = dff.topsw.to_numpy()
        topsw_cur.plot(x_pos,topsw, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black", linestyle="--")
        # Plot tops
        tops = dff.tops.to_numpy()
        tops_cur.plot(x_pos,tops, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black", linestyle="--")
        # Plot topsmm2
        topsmm2 = dff.topsmm2.to_numpy()
        topsmm2_cur.plot(x_pos,topsmm2, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black", linestyle="--")
        # Plot cf
        cf = dff.t_cf.to_numpy()
        cf_cur.plot(x_pos,cf, label=a, color=colors[ii_a], marker=markers[ii_a], markeredgecolor="black", linestyle="--")

    # configuration
    for i in range(0, 2):
        for j in range(0, 2):
            axs[i][j].set_xticks(x_pos, labels, rotation=45)
            axs[i][j].legend(loc="upper left")
    axs[0][0].set_ylabel(f"TOP/s/W ({workload})", fontsize=font_size)
    axs[0][1].set_ylabel(f"TOP/s ({workload})", fontsize=font_size)
    axs[1][0].set_ylabel(f"TOP/s/mm$^2$ ({workload})", fontsize=font_size)
    axs[1][1].set_ylabel(f"g, CO$_2$/Inference ({workload})", fontsize=font_size)
    # plt.text(0.5, -0.1, f"IMC size (#rows $\\times$ #cols)", ha="center", va="center", fontsize=20)  # does not work well
    plt.tight_layout()
    plt.show()

def plot_bar(i_df, imc_types, workload):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    width = 0.35
    font_size = 15
    df = i_df[i_df.workload == workload]
    df["imc"] = df.dim.astype(str) + f"$\\times$" + df.dim.astype(str)

    assert workload != "geo", f"geo has no cost breakdown"

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    e_bar = axs[0, 0]  # energy: pJ/inference (macro-only pJ/op, if workload = peak)
    l_bar = axs[0, 1]  # latency: ns/inference (=tclk, if workload = peak)
    a_bar = axs[1, 0]  # area: mm2 (macro area, if workload = peak)
    cf_bar = axs[1, 1]  # CF/inference (macro-only g, CO2/op, if workload = peak)

    for ii_a, a in enumerate(imc_types):
        if ii_a == 0:
            ii_pos = -1
        else:
            ii_pos = 1

        dff = df[df.imc_type == a]
        labels = dff.imc.to_numpy()  # x label
        # Create positions for the bars on the x-axis
        x_pos = np.arange(len(labels))
        # Plot the bars
        # Plot en bd
        en_bd = dff.en  # series: nJ
        base = [0 for x in range(0, len(labels))]
        i = 0
        for key in en_bd.iloc[0].keys():
            if ii_a == 0:
                val_type = key  # legend
            else:
                val_type = None
            val = np.array([x[key] for x in en_bd])
            e_bar.bar(x_pos+width/2*ii_pos, val, width, label=val_type, bottom = base, color = colors[i], edgecolor="black")
            # e_bar.bar(x_pos+width/2*ii_pos, val, width, bottom = base, color = colors[i], edgecolor="black")
            base = base + val
            i += 1
            if i >= len(colors):
                i = 0

        # Plot time bd
        lat_bd = dff.lat  # cycles
        tclk_total = dff.t_tclk  # ns
        base = [0 for x in range(0, len(labels))]
        if workload == "peak":
            ans = dff.tclk  # ns
        else:
            ans = lat_bd
        for key in ans.iloc[0].keys():
            if ii_a == 0:
                val_type = key  # legend
            else:
                val_type = None
            val = np.array([x[key] for x in ans])
            if workload != "peak":
                val = val * tclk_total  # ns
            l_bar.bar(x_pos+width/2*ii_pos, val, width, label=val_type, bottom = base, color = colors[i], edgecolor="black")
            base = base + val
            i += 1
            if i >= len(colors):
                i = 0

        # Plot area bd
        area_bd = dff.area  # mm2
        base = [0 for x in range(0, len(labels))]
        ans = area_bd
        for key in ans.iloc[0].keys():
            if ii_a == 0:
                val_type = key  # legend
            else:
                val_type = None
            val = np.array([x[key] for x in ans])  # mm2
            a_bar.bar(x_pos+width/2*ii_pos, val, width, label=val_type, bottom = base, color = colors[i], edgecolor="black")
            base = base + val
            i += 1
            if i >= len(colors):
                i = 0

        # Plot cf bd
        cf_bd = dff.cf  # g, CO2
        base = [0 for x in range(0, len(labels))]
        ans = cf_bd
        for key in ans.iloc[0].keys():
            if key == "ops":
                continue
            if ii_a == 0:
                val_type = key  # legend
            else:
                val_type = None
            val = np.array([x[key] for x in ans])  # g, CO2
            cf_bar.bar(x_pos+width/2*ii_pos, val, width, label=val_type, bottom = base, color = colors[i], edgecolor="black")
            base = base + val
            i += 1
            if i >= len(colors):
                i = 0

    # configuration
    for i in range(0, 2):
        for j in range(0, 2):
            axs[i][j].set_xticks(x_pos, labels, rotation=45)
            axs[i][j].legend(loc="upper left")
    if workload != "peak":
        axs[0][0].set_ylabel(f"pJ/Inference ({workload})", fontsize=font_size)
        axs[0][1].set_ylabel(f"ns/Inference ({workload})", fontsize=font_size)
        axs[1][0].set_ylabel(f"Area (mm$^2$) ({workload})", fontsize=font_size)
        axs[1][1].set_ylabel(f"g, CO$_2$/Inference ({workload})", fontsize=font_size)
    else:
        axs[0][0].set_ylabel(f"pJ/op ({workload})", fontsize=font_size)
        axs[0][1].set_ylabel(f"Tclk (ns) ({workload})", fontsize=font_size)
        axs[1][0].set_ylabel(f"Area (mm$^2$) ({workload})", fontsize=font_size)
        axs[1][1].set_ylabel(f"g, CO$_2$/Op ({workload})", fontsize=font_size)
    # plt.text(0.5, -0.1, f"IMC size (#rows $\\times$ #cols)", ha="center", va="center", fontsize=20)  # does not work well
    plt.tight_layout()
    plt.show()

        
def get_accelerator(imc_type, tech_param, hd_param, dims):

    assert imc_type in ["AIMC", "DIMC"], f"imc_type {imc_type} not in [AIMC, DIMC]"

    imc_array = ImcArray(tech_param, hd_param, dims)
    mem_hier = memory_hierarchy_dut(imc_array)
    core = {Core(1, imc_array, mem_hier)}
    acc_name = os.path.basename(__file__)[:-3]
    accelerator = Accelerator(acc_name, core)
    return accelerator

def get_param_setting(imc_type="DIMC", cols=32, rows=32, D3=1):
    # type: DIMC or AIMC
    # cols: int, can divide with 8
    # rows: int
    # D3: int

    ##################
    ## tech_param
    tech_param_28nm = {
        "tech_node":0.028,              # unit: um
        "vdd":      0.9,                # unit: V
        "nd2_cap":  0.7/1e3,            # unit: pF
        "xor2_cap": 0.7*1.5/1e3,        # unit: pF
        "dff_cap":  0.7*3/1e3,          # unit: pF
        "nd2_area": 0.614/1e6,          # unit: mm^2
        "xor2_area":0.614*2.4/1e6,      # unit: mm^2
        "dff_area": 0.614*6/1e6,        # unit: mm^2
        "nd2_dly":  0.0478,             # unit: ns
        "xor2_dly": 0.0478*2.4,         # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }

    ##################
    ## dimensions
    assert cols//8 == cols/8, f"cols {cols} cannot divide with 8."
    dimensions = {
        "D1": cols/8, # wordline dimension
        "D2": rows,   # bitline dimension
        "D3": D3,    # nb_macros
    }  # {"D1": ("K", 4), "D2": ("C", 32),}

    ##################
    ## hd_param
    assert imc_type in ["DIMC", "AIMC"], f"imc_type {imc_type} not in range [DIMC, AIMC]"
    if imc_type == "DIMC":
        imc = "digital"
        in_bits_per_cycle = 1
    else:
        imc = "analog"
        in_bits_per_cycle = 2
    hd_param = {
        "pe_type": "in_sram_computing", # for in-memory-computing. Digital core for different values.
        "imc_type": imc,                # "digital" or "analog"
        "input_precision":      8,      # activation precision
        "weight_precision":     8,      # weight precision
        "input_bit_per_cycle":  in_bits_per_cycle,  # nb_bits of input/cycle (treated as DAC resolution)
        "group_depth":          1,      # m factor
        "adc_resolution":       8,      # ADC resolution
        "wordline_dimension":   "D1",   # hardware dimension where wordline is (corresponds to the served dimension of input regs)
        "bitline_dimension":    "D2",   # hardware dimension where bitline is (corresponds to the served dimension of output regs)
        "enable_cacti":         True,   # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
        # Energy of writing weight. Required when enable_cacti is False.
        # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
    }
    hd_param["adc_resolution"] = hd_param["input_bit_per_cycle"] + 0.5 * int(math.log2(dimensions["D2"]))
    ##################
    ## return para
    return tech_param_28nm, hd_param, dimensions

def calc_cf(energy, lat, area, nb_of_ops, lifetime=3, chip_yield=0.95):
    # Used model: ACT
    # Use scenario: fixed-time
    # Ref: Gupta, Udit, et al. "ACT: Designing sustainable computer systems with an architectural carbon modeling tool." Proceedings of the 49th Annual International Symposium on Computer Architecture. 2022.
    # energy: unit: pJ
    # lat: unit: ns
    # area: unit: mm2
    # lifetime: unit: year
    # chip_yield: 0-1
    assert 0<chip_yield<=1, f"yield {chip_yield} is not in range (0,1]"
    #################
    ## para calc (specific for 28 nm)
    # unit: 
    # OP_CF -> g, CO2 per op from operational footprint
    # E_SOC -> g, CO2 per op from processors
    # E_CF = E_SOC (no off-chip mem considered)
    # TOTAL_CF -> g, CO2 in total per task
    OP_CF = 301 * energy / (3.6 * (10**18)) / nb_of_ops # g, CO2/kWh * energy (pJ) / 3.6E18 / nb_of_ops

    runtime = lat  # unit: ns
    scale_factor = runtime / (lifetime * (3.1536 * 10**16)) / nb_of_ops  # ns / (yr * 3.1536E16) / nb_of_ops

    E_SOC_EPA = 1/chip_yield * (301 * 0.9 /100) * area * scale_factor
    E_SOC_GPA = 1/chip_yield * (100 /100) * area * scale_factor
    E_SOC_MPA = 1/chip_yield * (500 /100) * area * scale_factor
    E_SOC = E_SOC_EPA + E_SOC_GPA + E_SOC_MPA  # g, CO2/op
    E_CF = E_SOC

    CF_PER_OP = OP_CF + E_CF  # g, CO2/op
    tt_cf_bd = {  # total cf breakdown details
            "ops":  nb_of_ops,
            "opcf": OP_CF,
            "soc_epa": E_SOC_EPA,
            "soc_gpa": E_SOC_GPA,
            "soc_mpa": E_SOC_MPA
            }
    return CF_PER_OP, tt_cf_bd


if __name__ == "__main__":
    #########################################
    ## parameter setting
    # workload: peak & ae from MLPerf Tiny
    # varibles: D1 => [32, 64, .., 1024]
    # varibles: D2 => D2 = D1
    # varibles: 
    Dimensions = [2**x for x in range(5, 11)]
    workloads = ["ae", "ds_cnn", "mobilenet", "resnet8", "peak"]  # peak: macro-level peak
    imc_types = ["AIMC", "DIMC"]
    ops_workloads = {'ae': 532512, 'ds_cnn': 5609536, 'mobilenet': 15907840, 'resnet8': 25302272}
    pickle_exsit = True  # read output directly if the output is saved in the last run

    if pickle_exsit == False:
        #########################################
        ## Simulation
        data_vals = []
        os.system("rm -rf outputs/*")
        for workload in workloads:
            for imc_type in imc_types:
                for d in Dimensions:
                    tech_param, hd_param, dims = get_param_setting(imc_type=imc_type, cols=d, rows=d, D3=1)
                    if workload == "peak":
                        imc = ImcArray(tech_param, hd_param, dims)
                        area_bd = imc.area_breakdown  # dict
                        area_total = imc.total_area  # float (mm2)
                        tclk_bd = imc.tclk_breakdown  # dict
                        tclk_total = imc.tclk  # float (ns)
                        peak_en_bd = imc.unit.get_peak_energy_single_cycle()
                        peak_en_total = sum([v for v in peak_en_bd.values()])  # float (pJ)
                        nb_of_ops = np.prod([x for x in dims.values()]) * hd_param["input_bit_per_cycle"] / hd_param["input_precision"] * 2
                        cf_total, cf_bd = calc_cf(energy=peak_en_total, lat=tclk_total, area=area_total, nb_of_ops=nb_of_ops, lifetime=3, chip_yield=0.95)  # unit: g, CO2/MAC
                        
                        res = {
                                "workload": workload,
                                "imc_type": imc_type,
                                "dim": d,
                                "ops": nb_of_ops,
                                "area": area_bd,
                                "lat": 1,
                                "tclk": tclk_bd,
                                "en": peak_en_bd,
                                "cf": cf_bd,
                                "t_area": area_total,
                                "t_lat": 1,
                                "t_tclk": tclk_total,
                                "t_en": peak_en_total,
                                "t_cf": cf_total,
                                "cme": None,
                                }
                        data_vals.append(res)
                    else:
                        if workload == "ds_cnn":
                            workload_dir = "zigzag/inputs/examples/workload/mlperf_tiny/ds_cnn.onnx"
                        elif workload == "ae":
                            workload_dir = "zigzag/inputs/examples/workload/mlperf_tiny/deepautoencoder.onnx"
                        elif workload == "mobilenet":
                            workload_dir = "zigzag/inputs/examples/workload/mlperf_tiny/mobilenet_v1.onnx"
                        elif workload == "resnet8":
                            workload_dir = "zigzag/inputs/examples/workload/mlperf_tiny/resnet8.onnx"
                        else:
                            breakpoint()  # to be filled
                        
                        mapping = "zigzag.inputs.examples.mapping.default_imc"
                        accelerator = get_accelerator(imc_type, tech_param, hd_param, dims)

                        # Call API
                        hw_name = imc_type
                        wl_name = re.split(r"/|\.", workload_dir)[-1]
                        if wl_name == "onnx":
                            wl_name = re.split(r"/|\.", workload_dir)[-2]
                        experiment_id = f"{hw_name}-{wl_name}"
                        pkl_name = f"{experiment_id}-saved_list_of_cmes"

                        ans = get_hardware_performance_zigzag(
                            workload_dir,
                            accelerator,
                            mapping,
                            opt="EDP",
                            dump_filename_pattern=f"outputs/{experiment_id}-layer_?.json",
                            pickle_filename=f"outputs/{pkl_name}.pickle",
                        )

                        # Read output
                        with open(f"outputs/{experiment_id}-layer_overall_complete.json", "r") as fp:
                            dat = json.load(fp)
                        en_total = dat["outputs"]["energy"]["energy_total"]  # float: pJ
                        lat_total = dat["outputs"]["latency"]["computation"]  # float: cycles
                        area_total = dat["outputs"]["area (mm^2)"]["total_area"]  # float: mm2
                        tclk_total = dat["outputs"]["clock"]["tclk (ns)"]  # float: ns
                        # breakdown
                        en_bd = {
                                "array": dat["outputs"]["energy"]["operational_energy"],
                                "mem": dat["outputs"]["energy"]["memory_energy"],
                                }
                        lat_bd = dat["outputs"]["latency"]["computation_breakdown"]
                        area_bd = dat["outputs"]["area (mm^2)"]["total_area_breakdown:"]
                        tclk_bd = dat["outputs"]["clock"]["tclk_breakdown (ns)"]
                        # calc CF (carbon footprint) (below is for g, CO2/op)
                        # cf_total, cf_bd = calc_cf(energy=en_total, lat=lat_total*tclk_total, area=area_total, nb_of_ops=ops_workloads[workload], lifetime=3, chip_yield=0.95)
                        # below is for g, CO2/inference
                        cf_total, cf_bd = calc_cf(energy=en_total, lat=lat_total*tclk_total, area=area_total, nb_of_ops=1, lifetime=3, chip_yield=0.95)
                        res = {
                                "workload": workload,
                                "imc_type": imc_type,
                                "dim": d,
                                "ops": ops_workloads[workload],
                                "area": area_bd,   # dict: mm2
                                "lat": lat_bd,     # dict: cycles
                                "tclk": tclk_bd,   # dict: ns
                                "en": en_bd,       # dict: pJ
                                "cf": cf_bd,       # dict: g, CO2
                                "t_area": area_total,
                                "t_lat": lat_total,
                                "t_tclk": tclk_total,
                                "t_en": en_total,
                                "t_cf": cf_total,
                                "cme": ans[2],
                        }
                        data_vals.append(res)

        df = pd.DataFrame(data_vals)

        #########################################
        ## Add geo-mean and calculation for topsw/tops/topsmm2
        data_vals = []
        for imc_type in imc_types:
            for dim in Dimensions:
                geo_topsw = 1
                geo_tops = 1
                geo_topsmm2 = 1
                geo_ops = 1
                geo_lat = 1
                geo_en = 1
                geo_cf = 1
                for workload in workloads:
                    if workload == "geo":  # skip if it's for geo-mean (will be re-calculated)
                        continue
                    dff = df[(df.workload == workload) & (df.imc_type == imc_type) & (df.dim == dim)]
                    new_res = {
                            "workload": dff.workload.to_list()[0],
                            "imc_type": dff.imc_type.to_list()[0],
                            "dim": dff.dim.to_list()[0],
                            "ops": dff.ops.to_list()[0],
                            "area": dff.area.to_list()[0],
                            "lat": dff.lat.to_list()[0],
                            "tclk": dff.tclk.to_list()[0],
                            "en": dff.en.to_list()[0],
                            "cf": dff.cf.to_list()[0],
                            "t_area": dff.t_area.to_list()[0],
                            "t_lat": dff.t_lat.to_list()[0],
                            "t_tclk": dff.t_tclk.to_list()[0],
                            "t_en": dff.t_en.to_list()[0],
                            "t_cf": dff.t_cf.to_list()[0],
                            "topsw": 1/(dff.t_en.to_list()[0] / dff.ops.to_list()[0]),  # 1/(pJ/op)
                            "tops": dff.ops.to_list()[0] / (dff.t_lat.to_list()[0] * dff.t_tclk.to_list()[0] * 1000),  # ops/ps
                            "topsmm2": dff.ops.to_list()[0] / (dff.t_lat.to_list()[0] * dff.t_tclk.to_list()[0] * 1000) / dff.t_area.to_list()[0],
                            "cme": dff.cme.to_list()[0],
                            }
                    if workload not in ["peak", "geo"]:
                        geo_topsw *= new_res["topsw"]
                        geo_tops *= new_res["tops"]
                        geo_topsmm2 *= new_res["topsmm2"]
                        geo_ops *= new_res["ops"]
                        geo_lat *= new_res["t_lat"]
                        geo_en *= new_res["t_en"]
                        geo_cf *= new_res["t_cf"]
                    data_vals.append(new_res)
                geo_topsw = geo_topsw ** (1/len(workloads))
                geo_tops = geo_tops ** (1/len(workloads))
                geo_topsmm2 = geo_topsmm2 ** (1/len(workloads))
                geo_res = {
                        "workload": "geo",
                        "imc_type": imc_type,
                        "dim": dim,
                        "ops": geo_ops,
                        "area": new_res["area"],
                        "lat": None,
                        "tclk": new_res["tclk"],
                        "en": None,
                        "cf": None,
                        "t_area": new_res["t_area"],
                        "t_lat": geo_lat,
                        "t_tclk": new_res["t_tclk"],
                        "t_en": geo_en,
                        "t_cf": geo_cf,
                        "topsw": geo_topsw,
                        "tops": geo_tops,
                        "topsmm2": geo_topsmm2,
                        }
                data_vals.append(geo_res)
        new_df = pd.DataFrame(data_vals)
        df = new_df

        # save df to pickle
        with open("expr_res.pkl", "wb") as fp:
            pickle.dump(df, fp)
    else:
        # load df from pickle
        with open("expr_res.pkl", "rb") as fp:
            df = pickle.load(fp)

    #########################################
    ## Visualization
    plot_bar(i_df=df, imc_types=imc_types, workload="ds_cnn")
    ## The color of the plot has not been fixed when workload == "peak". Now the color display is in a mess order.
    ## The cause if the elements in AIMC and DIMC are different to each other.
    # plot_curve(i_df=df, imc_types=imc_types, workload="ds_cnn")
