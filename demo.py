## create demo file for master project proposal

import tkinter as tk
from tkinter import ttk, PhotoImage, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from expr import read_pickle
from PIL import ImageTk, Image

class demo:
    def __init__(self):
        # Read in parameters
        df = read_pickle("no_cme_expr_res_tiny.pkl")
        workloads = ["ae", "ds_cnn", "mobilenet", "resnet8"]  # legal workload keywords
        periods = {"peak": 1, "ae": 1e+9 / 1, "ds_cnn": 1e+9 / 10, "mobilenet": 1e+9 / 0.75, "resnet8": 1e+9 / 25,
                   "geo": 1e+9 / ((1 * 10 * 0.75 * 25) ** 0.25),
                   "resnet18": 1e+9 / 25,
                   "deeplabv3": 1e+9 / 25, "mobilebert": 1e+9 / 25, "mobilenet_edgetpu": 1e+9 / 25,
                   "mobilenet_v2": 1e+9 / 25, }
        raw_data = {"data": df, "workloads": workloads, "periods": periods, }
        acc_types = ["pdigital_os", "DIMC"]
        workload = "geo"
        sram_sizes = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 1024 * 1024]  # unit: B

        self.df = df
        self.periods = periods
        self.raw_data = raw_data
        self.acc_types = acc_types
        self.sram_sizes = sram_sizes
        self.workload = workload

        # Create main application window
        self.root = tk.Tk()
        self.root.title('Carbon cost design space exploration')

        # Create a frame for the plot
        frame = ttk.Frame(self.root)
        # frame.pack_propagate(False)  # Prevent frame from resizing to fit widgets
        frame.pack(padx=15, pady=15)

        # Create a matplotlib figure and axis
        fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(fig, master=frame)  # Embed figure in Tkinter window
        self.canvas.get_tk_widget().pack()

        # Create an entry widget for user input: acc
        entry_label = ttk.Label(self.root, text="Enter acc type (pdigital_ws, pdigital_os, DIMC, AIMC):")
        entry_label.pack()
        self.acc_entry = ttk.Entry(self.root)
        self.acc_entry.insert(0, "DIMC")
        self.acc_entry.pack()

        # Create an entry widget for user input (y axis): carbon or topsw
        entry_label_y = ttk.Label(self.root, text="Enter y axis (area, carbon, topsw, carbon_bd):")
        entry_label_y.pack()
        self.y_entry = ttk.Entry(self.root)
        self.y_entry.insert(0, "topsw")
        self.y_entry.pack()

        # Create an entry widget for user input (x axis): carbon or topsw
        entry_label_x = ttk.Label(self.root, text="Enter x axis (area, carbon, topsw):")
        entry_label_x.pack()
        self.x_entry = ttk.Entry(self.root)
        self.x_entry.insert(0, "area")
        self.x_entry.pack()

        # Create a button to update the plot
        update_button = ttk.Button(self.root, text="Update Plot", command=self.update_plot)
        update_button.pack()
        exit_button = ttk.Button(self.root, text="Exit", command=self.exit_gui)
        exit_button.pack()

        # Initial plot
        self.update_plot()

        # Run the GUI event loop
        self.root.mainloop()

    def update_plot(self):
        # Clear previous plot
        self.ax.clear()

        acc_input = self.acc_entry.get().split(", ")
        for acc in acc_input:
            if acc not in ["pdigital_ws", "pdigital_os", "DIMC", "AIMC"]:
                return -1
        self.acc_types = acc_input

        if self.y_entry.get() not in ["carbon", "topsw", "carbon_bd", "area"]:
            return -1
        else:
            y_input = self.y_entry.get()

        if self.x_entry.get() not in ["carbon", "topsw", "area"]:
            return -1
        else:
            x_input = self.x_entry.get()

        colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
                  u'#bcbd22', u'#17becf']
        markers = ["s", "o", "^", "p", "D", "P"]
        df = self.raw_data["data"]
        df = df[(df.workload == self.workload)]
        periods = self.raw_data["periods"]
        marker_size = 50
        for ii_a, a in enumerate(self.acc_types):
            for ii_b, sram_size in enumerate(self.sram_sizes):
                dims = sorted(list(set(df["dim"].to_list())))
                for ii_c, dim in enumerate(dims):
                    dff = df[(df.acc_type == a) & (df.sram_size == sram_size) & (df.dim == dim)]
                    # Create positions for the bars on the x-axis
                    area = dff.t_area.to_numpy()

                    # Plot cf, simple (fixed-time)
                    topsw = dff.topsw.to_numpy()
                    cfft = dff.t_cf_ft_ex_pkg.to_numpy()
                    cffw = dff.t_cf_fw_ex_pkg.to_numpy()
                    tmp = self.raw_data["data"]
                    geo_cfft_op = 1
                    geo_cfft_em = 1
                    geo_cffw_op = 1
                    geo_cffw_em = 1
                    for workload in ["ae", "ds_cnn", "resnet8", "mobilenet"]:
                        tmp_wk = tmp[(tmp.workload == workload)]
                        tmp_wk = tmp_wk[(tmp_wk.acc_type == a) & (tmp_wk.sram_size == sram_size) & (tmp_wk.dim == dim)]
                        cfft_op = np.array([x["opcf"] for x in tmp_wk.cf_ft])
                        cfft_em = np.array([x["soc_epa"] + x["soc_gpa"] + x["soc_mpa"] + x["dram"] for x in tmp_wk.cf_ft])
                        cffw_op = np.array([x["opcf"] for x in tmp_wk.cf_fw])
                        cffw_em = np.array([x["soc_epa"] + x["soc_gpa"] + x["soc_mpa"] + x["dram"] for x in tmp_wk.cf_fw])
                        geo_cfft_op *= cfft_op
                        geo_cfft_em *= cfft_em
                        geo_cffw_op *= cffw_op
                        geo_cffw_em *= cffw_em
                    geo_cfft_op = geo_cfft_op ** 0.25
                    geo_cfft_em = geo_cfft_em ** 0.25
                    geo_cffw_op = geo_cffw_op ** 0.25
                    geo_cffw_em = geo_cffw_em ** 0.25

                    if len(acc_input) == 1:
                        if ii_c == 0:
                            mem_kb = sram_size//1024
                            if mem_kb >= 1024:
                                mem_mb = mem_kb // 1024
                                label = f"{a}, {mem_mb}MB SRAM"
                            else:
                                mem_mb = None
                                label = f"{a}, {mem_kb}KB SRAM"
                        else:
                            label = None
                    else:
                        if ii_b == 0 and ii_c == 0:
                            label = f"{a}"
                        else:
                            label = None

                    if x_input == "carbon":
                        x_bar = cfft
                    elif x_input == "area":
                        x_bar = area
                    elif x_input == "topsw":
                        x_bar = topsw
                    else:
                        pass

                    if y_input == "carbon":
                        self.ax.scatter(x_bar, cfft, label=label,
                                        color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
                    elif y_input == "topsw":
                        self.ax.scatter(x_bar, topsw, label=label,
                                        color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
                    elif y_input == "area":
                        self.ax.scatter(x_bar, area, label=label,
                                        color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
                    elif y_input == "carbon_bd":
                        self.ax.scatter(x_bar, geo_cfft_op, label=label,
                                        color="white", marker=markers[ii_a], edgecolors=colors[ii_b], s=marker_size)
                        self.ax.scatter(x_bar, geo_cfft_em,
                                        color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
                    else:
                        pass

                    # topsw_scatter.scatter(area, topsw,
                    # topsw_scatter.scatter(topsw, cfft,
                    # ax.scatter(topsw, geo_cffw_op,
                    #                       color="white", marker=markers[ii_a], edgecolors=colors[ii_b], s=marker_size)
                    # self.ax.scatter(topsw, cfft,
                    #                       color=colors[ii_b], marker=markers[ii_a], edgecolors="black", s=marker_size)
        # configuration
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.grid(which="both")
        self.ax.set_axisbelow(True)
        self.ax.legend()
        if x_input == "area":
            self.ax.set_xlabel(f"Area (mm2)")
        elif x_input == "topsw":
            self.ax.set_xlabel(f"TOP/s/W")
        elif x_input == "carbon":
            self.ax.set_xlabel(f"g, CO2")
        else:
            self.ax.set_xlabel(f"")
        if y_input in ["carbon", "carbon_bd"]:
            self.ax.set_ylabel(f"g, CO2")
        elif y_input in ["topsw"]:
            self.ax.set_ylabel(f"TOP/s/W")
        elif y_input == "area":
            self.ax.set_ylabel(f"Area (mm2)")
        else:
            self.ax.set_ylabel("")

        # Update canvas
        self.canvas.draw()

    def exit_gui(self):
        self.root.destroy()


if __name__ == "__main__":
    demo()

