import os
import random
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.get_cacti_cost import get_w_cost_per_weight_from_cacti
from zigzag.classes.hardware.architecture.get_cacti_cost import get_cacti_cost
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core

from zigzag.api import *
import pickle

def memory_hierarchy_dut(parray, visualize=False, sram_size=64*1024):
    # This function defines the memory hierarchy in the hardware template.
    # @para parray: digital pe array object
    # @para visualize: whether illustrate teh memory hierarchy
    # @para sram_size: define the on-chip sram size, unit: byte
    """ [OPTIONAL] Get w_cost of imc cell group from CACTI if required """
    cacti_path = "zigzag/classes/cacti/cacti_master"

    """Memory hierarchy variables"""
    """ size=#bit, bw=(read bw, write bw), cost=(read word energy, write work energy) """
    reg_W1 = MemoryInstance(
        name="reg_W1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # unit: pJ/weight
        area=0.614 * 6 / 1e6 * 8,
        r_port=1,  # 1 standalone read port
        w_port=1,  # 1 standalone write port
        rw_port=0,  # no port for both reading and writing
        latency=1,  # 1 extra clock cycle required
    )
    reg_I1 = MemoryInstance(
        name="rf_I1",
        size=8,
        r_bw=8,
        w_bw=8,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 8,  # pJ/access
        area=0.614 * 6 / 1e6 * 8,  # mm^2
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )

    reg_O1 = MemoryInstance(
        name="rf_O1",
        size=16,
        r_bw=16,
        w_bw=16,
        r_cost=0,
        w_cost=0.7 * 3 / 1e3 * (0.9**2) * 16, # pJ/access
        area=0.614 * 6 / 1e6 * 16, # mm^2
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )

    ##################################### on-chip memory hierarchy building blocks #####################################

    # sram_size = 256 * 1024 # unit: byte
    # dimension_sizes:
    # [0]: D1, [1]: D2, [2]: D3
    sram_bw = max(parray.dimension_sizes[1] * 8 * parray.dimension_sizes[2],
                  parray.dimension_sizes[0] * 16 * parray.dimension_sizes[2])

    ac_time, sram_area, sram_r_cost, sram_w_cost = get_cacti_cost(cacti_path, 0.028, "sram", sram_size, sram_bw, hd_hash=str(hash((sram_size, sram_bw, random.randbytes(8)))))

    sram_256KB_256_3r_3w = MemoryInstance(
        name="sram_256KB",
        size=sram_size * 8, # byte -> bit
        r_bw=sram_bw,
        w_bw=sram_bw,
        r_cost=sram_r_cost,
        w_cost=sram_w_cost,
        area=sram_area,
        r_port=4,
        w_port=4,
        rw_port=0,
        latency=1,
        min_r_granularity=sram_bw//16, # assume there are 16 sub-banks
        min_w_granularity=sram_bw//16, # assume there are 16 sub-banks
    )

    #######################################################################################################################

    # dram_size = 1*1024*1024*1024 # unit: byte
    dram_size = 1 * 1024 * 1024  # unit: byte (change to 1MB to fit for carbon estimation for tinyml perf workloads)
    dram_ac_cost_per_bit = 3.7 # unit: pJ/bit
    dram_bw = parray.dimension_sizes[0] * 8 * parray.dimension_sizes[2]
    dram_100MB_32_3r_3w = MemoryInstance(
        name="dram_1GB",
        size=dram_size*8, # byte -> bit
        r_bw=dram_bw,
        w_bw=dram_bw,
        r_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        w_cost=dram_ac_cost_per_bit*dram_bw, # pJ/access
        area=0,
        r_port=3,
        w_port=3,
        rw_port=0,
        latency=1,
        min_r_granularity=dram_bw // 16,  # assume there are 16 sub-banks
        min_w_granularity=dram_bw // 16,  # assume there are 16 sub-banks
    )

    memory_hierarchy_graph = MemoryHierarchy(operational_array=parray)

    """
    fh: from high = wr_in_by_high 
    fl: from low = wr_in_by_low 
    th: to high = rd_out_to_high
    tl: to low = rd_out_to_low
    """
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_W1,
        operands=("I2",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions=set(),
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_I1,
        operands=("I1",),
        port_alloc=({"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},),
        served_dimensions={(1, 0, 0)},
    )
    memory_hierarchy_graph.add_memory(
        memory_instance=reg_O1,
        operands=("O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_2", "th": "r_port_2"},),
        served_dimensions={(0, 1, 0)},
    )

    ##################################### on-chip highest memory hierarchy initialization #####################################

    memory_hierarchy_graph.add_memory(
        memory_instance=sram_256KB_256_3r_3w,
        operands=("I1", "I2", "O",),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions="all",
    )

    ####################################################################################################################

    memory_hierarchy_graph.add_memory(
        memory_instance=dram_100MB_32_3r_3w,
        operands=("I1", "I2", "O"),
        port_alloc=(
            {"fh": "w_port_1", "tl": "r_port_1", "fl": None, "th": None},
            {"fh": "w_port_2", "tl": "r_port_2", "fl": None, "th": None},
            {"fh": "w_port_1", "tl": "r_port_1", "fl": "w_port_3", "th": "r_port_3"},
        ),
        served_dimensions="all",
    )

    if visualize:
        from zigzag.visualization.graph.memory_hierarchy import (
            visualize_memory_hierarchy_graph,
        )

        visualize_memory_hierarchy_graph(memory_hierarchy_graph)
    return memory_hierarchy_graph

def multiplier_array_dut():
    # Data is from envision paper (https://ieeexplore.ieee.org/abstract/document/7870353)
    # Tech: 28nm UTBB FD-SOI
    # PE array area is estimated from the chip picture, with total size: 0.75mm x 1.29mm for 16 x 16 int16 PEs
    # TOP/s/w: 2
    # Supply voltage: 1 V
    # Clock freq: 200 MHz
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.25  # unit: pJ/mac
    multiplier_area = 0.75*1.29/256/4  # mm2/PE
    dimensions = {
            "D1": 1024,
            "D2": 1024,
            "D3": 1,
    }
    multiplier = Multiplier(
        multiplier_input_precision, multiplier_energy, multiplier_area
    )
    multiplier_array = MultiplierArray(multiplier, dimensions)

    return multiplier_array

def cores_dut():
    multiplier_array1 = multiplier_array_dut()
    memory_hierarchy1 = memory_hierarchy_dut(multiplier_array1, sram_size=64*1024, visualize=True)

    core1 = Core(1, multiplier_array1, memory_hierarchy1)

    return {core1}

if __name__ == "__main__":
        cores = cores_dut()
        acc_name = os.path.basename(__file__)[:-3]
        accelerator = Accelerator(acc_name, cores)

        workload = "zigzag/inputs/examples/workload/mlperf_tiny/deepautoencoder.onnx"
        mapping = "zigzag.inputs.examples.mapping.default_imc"

        (energy, latency, cmes) = get_hardware_performance_zigzag(
                workload, accelerator, mapping)
        breakpoint()
