from zigzag.classes.stages.Stage import Stage

import networkx as nx
from typing import Generator
from zigzag.classes.workload.dummy_node import DummyNode

import logging

logger = logging.getLogger(__name__)

#################### Description ####################
## This stage must be processed before WorkloadStage.
## This stage figures out the no-use top memory levels for "I", "W", "O" when the size of lower memory level is enough to hold all data, considering the output data of previous layer can be directly used by next layer. As an impact, the energy / latency related to these memories will be removed.
## The general criteria is:
##      If a low-level memory size is big enough to hold both "I" and "O" data of current layer, memory above this one will be labeled as no-use.
##      If a low-level memory size is big enough to hold "W" data of entire workload, memory above this one will be labeled as no-use.
## The above method only applies layers along the same branch, otherwise (for branch starting nodes or branch final nodes) the "O" data will return back to the top possible memory.
## In RemoveNoUseMemStage, no-use mem across all layers, labeled in this stage, will be removed in the memory architecture.
## For now, the number of cores must be 1.
#################### Pseudo-code ####################
## Initialization:
##   mem_update_list = [layer_ids: {"I" / "O": -1}] ## mem level of different operands of each layer (there should be no -1 after self.update_top_mem_level())
##   each_layer_IO_data_size = [layer_ids: {"I" / "O": size}] ## input / output data size of each layer
##   mem_update_weight = top_mem_level ## top mem level to put weight
##   weight_size_entire_workload = weight_size # weight data size of entire workload
## Generate:
##   layer_execution_order = list( topological_sort(layer_gragh) )
## Locate top mem level for each operand of each layer. Store results in mem_update_list and mem_update_weight.
##   for layer in all_layers:
##     if layer.index != 0: ## not the 1st execution layer
##       mem_udpate_list[layer]["I"] = mem_udpate_list[previous_layer]["O"]
##     if len(layer.next_node) > 1 or len(next_layer.prevous_node) > 1: ## starting node of branches / final node of branches
##     | if layer.index == 0:
##     |   mem_update_list[layer]["I" / "O"] updates to the top input/output mem level
##     | else:
##     |   mem_update_list[layer]["O"] updates to the top output mem level
##     |   mem_update_weight = top weight mem level, if mem_update_weight > top weight mem level
##     |
##     else:
##       for mem in mem_levels(sort_order: from top to bottom):
##         if sum(layer[operand_size] for operand in mem.operands) <= mem.size:
##           if ["I", "O"] both in mem.operands:
##             mem_update_list[layer]["O"] = current_mem_level
##             if layer.index == 0: ## the 1st execution layer
##               mem_update_list[layer]["I"] = current_mem_level
##           if ("W" in mem.operand) and (current_mem_level < mem_update_weight):
##             mem_update_weight = current_mem_level
#####################################################

class SearchNoUseMemStage(Stage):
    def __init__(self, list_of_callables, *, accelerator, workload, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.workload = workload
        ## Initialization
        self.mem_update_list = {}
        self.each_layer_IO_data_size = {} # unit: bit
        core_id = accelerator.cores[0].id # correct only for single-core hardware
        self.core_mem_level_list = accelerator.get_core(core_id=core_id).memory_hierarchy.mem_level_list
        self.mem_update_weight = len(self.core_mem_level_list)-1 # index of the top memory
        self.weight_size_entire_workload = 0 # unit: bit
        self.layer_list = {} # layer name and its corresponding id
        for id, layer in enumerate(nx.topological_sort(workload)):
            if type(layer) != DummyNode: # create record on memory level, data size of each operand for un-dummy nodes
                try:
                    self.mem_update_list[f"{id}"] = [{operand: -1} for operand in accelerator.get_core(core_id=core_id).mem_hierarchy_dict.keys() if operand != layer.memory_operand_links[layer.constant_operands[0]]]
                except:
                    pass
                self.each_layer_IO_data_size[f"{id}"] = [{layer.memory_operand_links[operand]: layer.operand_size_bit[operand] for operand in layer.memory_operand_links.keys() if operand not in layer.constant_operands}]
                self.weight_size_entire_workload += layer.operand_size_bit[layer.constant_operands[0]]
                self.layer_list[layer] = id

    def run(self, workload_data_always_from_top_mem=False) -> Generator:
        self.update_top_mem_level() # figure out lowest possible mem level for all operands for all layers

        if workload_data_always_from_top_mem:
            # [OPTIONAL] re-define the input/output mem level of first/last layer to the top possible mem level. This
            # is specially designed for the case that workload input and output must be stored in the top mem level.
            self.update_mem_level_for_loading_data()

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], 
                accelerator=self.accelerator, 
                workload=self.workload,
                mem_update_list=self.mem_update_list,
                mem_update_weight=self.mem_update_weight,
                layer_list=self.layer_list,
                **self.kwargs,)
        for cme, (layer, extra_info) in sub_stage.run():
            yield cme, (layer, extra_info)

    def update_top_mem_level(self):
        """
        Update mem_update_list and mem_update_weight according to the algorithm description at the file beginning.
        """
        """
        param const_operand: constant operand name (e.g. "W")
        param act_operand: activation operand name (e.g. "I")
        param output_operand: output operand name (e.g. "O")
        """
        self.remove_dummy_nodes_in_workload() # remove dummy nodes for the ease of telling the branch starting or final nodes

        ## Update mem_update_list and mem_update_weight
        for id, layer in enumerate(nx.topological_sort(self.workload)):
            branch_starting_node = True if self.workload.out_degree(layer) > 1 else False # starting node of branches
            branch_final_node = True if self.workload.out_degree(layer) == 1 and self.workload.in_degree(list(self.workload.successors(layer))[0]) > 1 else False
            const_operand = layer.memory_operand_links[layer.constant_operands[0]]  # weight representation
            act_operand = layer.memory_operand_links[[operand for operand in layer.input_operands if operand not in layer.constant_operands][0]]  # act representation
            output_operand = layer.output_operand  # output representation
            curr_id = self.layer_list[layer]  # current layer id (key) in mem_udpate_list
            if id != 0:  ## not the first layer
                ## Assign mem_udpate_list[layer]["I"] = mem_udpate_list[previous_layer]["O"]
                prev_layer = list(self.workload.predecessors(layer))[0]  # previous layer node (object)
                prev_layer_id = self.layer_list[prev_layer]  # previous layer id
                prev_layer_output_operand = prev_layer.output_operand  # output representation of previous layer
                for ele in self.mem_update_list[f"{prev_layer_id}"]:  # find the output mem level of previous layer
                    try:
                        prev_layer_output_level = ele[f"{prev_layer_output_operand}"]
                    except KeyError:  # skip if the key is incorrect, as there will only be one that match.
                        pass
                self.update_IO_mem_level(curr_id, act_operand, prev_layer_output_level) # update the input mem level of current layer
            if branch_starting_node or branch_final_node: ## branch starting node or branch final node
                ## Update input, weight, output mem level for branch starting node and branch final node
                ## Find the top mem level for input if it is the first layer, update mem_udpate_list of current layer
                if id==0: ## the first layer
                    for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                        served_operands = list(mem.mem_level_of_operands.keys()) # Check the served operand of current mem
                        if act_operand in served_operands:
                            self.update_IO_mem_level(curr_id, act_operand, curr_mem_level) # update the input mem level of current layer if it is the first layer
                            break
                ## Find the top mem level for output, update mem_udpate_list of current layer
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys()) # Check the served operand of current mem
                    if output_operand in served_operands:
                        self.update_IO_mem_level(curr_id, output_operand, curr_mem_level) # update the output mem level of current layer
                        break
                ## Find the top mem level for weight, update mem_update_weight of current layer to the top weight mem level if mem_update_weight is bigger
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys()) # Check the served operand of current mem
                    if const_operand in served_operands: # identify the top weight mem level
                        if curr_mem_level < self.mem_update_weight: # mem_update_weight is bigger than the top weight mem level
                            self.mem_update_weight = curr_mem_level
                        break
            else: ## node (layer) that is not a branch starting node or a branch final node
                ## Iterate the memory level and update input, weight, output mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys()) # Check the served operand of current mem
                    ## Update input, weight, output mem level
                    avail_mem_size = mem.memory_instance.size  # available hardware mem size
                    mem_serve_io_both = True if (act_operand in served_operands) and (output_operand in served_operands) else False # ["I", "O"] both in mem.served_operands
                    mem_serve_weight = True if (const_operand in served_operands) else False # mem.served_operands = ["W"]
                    if mem_serve_io_both or mem_serve_weight:
                        required_IO_data_size = sum([self.each_layer_IO_data_size[f"{curr_id}"][0][operand] for operand in served_operands if operand != const_operand])
                        required_weight_size = self.weight_size_entire_workload if const_operand in served_operands else 0
                        required_total_size = required_IO_data_size + required_weight_size # required size to put data in current mem level
                        if required_total_size <= avail_mem_size: # sum(layer[operand_size] for operand in mem.operands) <= mem.size
                            if mem_serve_io_both:
                                if id == 0:
                                    self.update_IO_mem_level(curr_id, act_operand, curr_mem_level) # update input mem level
                                self.update_IO_mem_level(curr_id, output_operand, curr_mem_level) # update output mem level
                            if (curr_mem_level < self.mem_update_weight) and mem_serve_weight:  # update weight mem level
                                self.mem_update_weight = curr_mem_level
        ## [OPTIONAL CHECK] assert check if there is -1 value in mem_update_list
        ## [NOTE] Until here, if there is still -1 value in mem_update_list, it means the size of top mem level for IO is not big enough.
        for layer_ele in self.mem_update_list.values():
            for operand_dict in layer_ele:
                assert list(operand_dict.values())[0] >= 0

    def update_mem_level_for_loading_data(self):
        """
        [OPTIONAL FUNCTION] This is an optional function.
        Depending on your requirement, sometimes data loading from the top mem level and offloading to the top mem level is a must.
        If that is the your case, add this function to self.run().
        Otherwise, if the input is generated on-chip at the lowest possible input mem level and the output is stored on-chip at the lowest possible output mem level, remove this function from self.run().
        [FUNCTION OBJECT]
        Update mem_update_list of first and last layer, so that the input data of first layer still is loaded from top input mem level and the output of last layer still is offloaded to top output mem level
        """
        self.remove_dummy_nodes_in_workload()  # remove dummy nodes for the ease of telling the branch starting or final nodes

        ## Update mem_update_list and mem_update_weight
        for id, layer in enumerate(nx.topological_sort(self.workload)):
            act_operand = layer.memory_operand_links[[operand for operand in layer.input_operands if operand not in layer.constant_operands][0]]  # act representation
            output_operand = layer.output_operand  # output representation
            curr_id = self.layer_list[layer]  # current layer id (key) in mem_udpate_list
            if id == 0: # the first layer: update activation mem level to the top possible mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if act_operand in served_operands:
                        self.update_IO_mem_level(curr_id, act_operand, curr_mem_level)  # update the input mem level of current layer if it is the first layer
                        break
            if id == len(self.layer_list) - 1: # the last layer: update output mem level to the top possible mem level
                for curr_mem_level, mem in reversed(list(enumerate(self.core_mem_level_list))):
                    served_operands = list(mem.mem_level_of_operands.keys())  # Check the served operand of current mem
                    if output_operand in served_operands:
                        self.update_IO_mem_level(curr_id, output_operand, curr_mem_level)  # update the output mem level of current layer if it is the last layer
                        break
    def remove_dummy_nodes_in_workload(self):
        ## Remove dummy nodes (layers) in the graph (assume there is no branch from a non-dummy node to dummy node)
        ## Redirect the outgoing edges of dummy nodes to non-dummy nodes
        ## Algorithm:
        ## for each dummy node, add edges between its predecessor nodes and successor nodes; then remove the dummy node.
        #############################################
        ## Comment on the following 4 lines below: visualize the network for debugging
        ## import matplotlib.pyplot as plt
        ## pos = nx.spring_layout(self.workload)
        ## nx.draw(self.workload, pos, with_labels=True, node_color="lightblue", font_weight="bold")
        ## plt.show()
        #############################################
        dummy_nodes = [node for node in self.workload.nodes() if type(node) == DummyNode]
        for dummy_node in dummy_nodes:
            for successor_node in list(self.workload.successors(dummy_node)):
                for predecessor_node in list(self.workload.predecessors(dummy_node)):
                    self.workload.add_edge(predecessor_node, successor_node)
        self.workload.remove_nodes_from(dummy_nodes)

    def update_IO_mem_level(self, layer_id, operand, target_level):
        """
        Update self.mem_update_list as:
        self.mem_update_list[layer_id][operand_index][operand] = target_level
        """
        for pos, ele in enumerate(self.mem_update_list[f"{layer_id}"]):
            if list(ele.keys())[0] == f"{operand}":
                self.mem_update_list[f"{layer_id}"][pos][f"{operand}"] = target_level