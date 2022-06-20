import pygraphviz as pgv
from itertools import product
from collections import defaultdict
from typing import Dict, Set
from IPython.display import Image, display
import opyenxes
import pm4py
import pandas as pd
import numpy as np

class MyGraph(pgv.AGraph):
 
    def __init__(self, *args):
        super(MyGraph, self).__init__(strict=False, directed=True, *args)
        self.graph_attr['rankdir'] = 'LR'
        self.node_attr['shape'] = 'Mrecord'
        self.graph_attr['splines'] = 'ortho'
        self.graph_attr['nodesep'] = '0.8'
        self.edge_attr.update(penwidth='2')
 
    def get_trace_from_csv(self, filename):
        df = pd.read_csv(filename)
        df = df.sort_values(by=['Case ID', 'Start Timestamp'])
        number_of_cases = df['Case ID'].max()
        trace = np.empty(number_of_cases, dtype=object)
        for i in range(trace.shape[0]):
            trace[i] = []
        df = df.reset_index()
        for index, row in df.iterrows():
            trace[row['Case ID']-1].append(row['Activity'])
        return trace

    def get_events_from_csv(self, filename):
        df = pd.read_csv(filename)
        df = df.sort_values(by=['Case ID', 'Start Timestamp'])
        return df['Activity'].unique()

    def get_trace_from_xes(self, filename):
      event_log = pm4py.read_xes(filename)
      data = []
      case_id = 1
      for trace in event_log:
        for event in trace:
          row = [case_id, event['Activity'], event['time:timestamp']]
          data.append(row)
        case_id += 1
      df = pd.DataFrame.from_records(data, columns=['Case ID', 'Activity', "Start Timestamp"])
      number_of_cases = df['Case ID'].max()
      trace = np.empty(number_of_cases, dtype=object)
      for i in range(trace.shape[0]):
        trace[i] = []
      df = df.reset_index()
      for index, row in df.iterrows():
        trace[row['Case ID']-1].append(row['Activity'])
      return trace

    def get_events_from_xes(self, filename):
      event_log = pm4py.read_xes(filename)
      data = []
      case_id = 1
      for trace in event_log:
        for event in trace:
          row = [case_id, event['Activity'], event['time:timestamp']]
          data.append(row)
        case_id += 1
      df = pd.DataFrame.from_records(data, columns=['Case ID', 'Activity', "Start Timestamp"])
      return df['Activity'].unique()

    def get_direct_succession(self, trace, events):
        direct_succession = defaultdict(set)
        for case in trace:
            for index, event in enumerate(case):
                if (index > 0):
                    direct_succession[case[index-1]].add(event)
        for event in events:
            if event not in direct_succession.keys():
                direct_succession[event] = set()
        return dict(direct_succession)
    
    def get_direct_succession_count(self, trace):
        direct_succession_count = defaultdict(int)
        for case in trace:
            for index, event in enumerate(case):
                if (index > 0):
                    if (direct_succession_count[case[index-1] + event] == None):
                        direct_succession_count[case[index-1] + event] = 1
                    else:
                        direct_succession_count[case[index-1] + event] += 1
        return dict(direct_succession_count)
    
    def filter_trace_lowerbound(self, trace, direct_succession_count, lowerbound):
        for i, case in enumerate(trace):
            removed = True
            for index, event in enumerate(case):
                if (index > 0):
                    if (direct_succession_count[case[index-1] + event] > lowerbound):
                        removed = False
            if (removed):
                trace[i] = None
        return trace

    def add_edge(self, source, target):
        super(MyGraph, self).add_edge(source, target)

    def add_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="")
 
    def add_end_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="",penwidth='3')

    def add_and_gateway(self, *args):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                  width=".7",height=".7",
                                  fixedsize="true",
                                  fontsize="40",label="+")
 
    def add_xor_gateway(self, *args, **kwargs):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                  width=".7",height=".7",
                                  fixedsize="true",
                                  fontsize="40",label="×")
        
    def add_source_edge(self, source, target, direct_succession, *args):
        if source is not "start" and source in direct_succession[source]:
            gateway = 'XORs '+str(source)+'->'+str(source) 
            self.add_xor_gateway(gateway, *args)
            super(MyGraph, self).add_edge(source, gateway)
            super(MyGraph, self).add_edge(gateway, source)
            super(MyGraph, self).add_edge(gateway, target)
        else:
            super(MyGraph, self).add_edge(source, target)

    def add_start_edge(self, source, target, direct_succession, start_set_events, *args):
        loop = []
        for start_event in start_set_events:
                for event in direct_succession[start_event]:
                    if len(direct_succession[event]) == 1 and list(direct_succession[event])[0] == start_event:
                        loop.append((start_event, event))
        if len(loop) > 0:
            for l in loop:
                gateway = 'XORs '+str(source)+'->{'+str(l[0])+', '+str(l[1])+'}' 
                self.add_xor_gateway(gateway, *args)
                super(MyGraph, self).add_edge(source, gateway)
                super(MyGraph, self).add_edge(gateway, l[0])
                super(MyGraph, self).add_edge(l[1], gateway)
        else:    
            super(MyGraph, self).add_edge(source, target)

    def add_target_edge(self, source, target, direct_succession, *args):
        loop = False #b,d
        for event in direct_succession[source]:
            if len(direct_succession[event]) == 1 and list(direct_succession[event])[0] == source:
                loop = True
                gateway = 'XORs '+str(source)+'->{'+str(target)+', '+str(event)+'}' 
                self.add_xor_gateway(gateway, *args)
                super(MyGraph, self).add_edge(source, gateway)
                super(MyGraph, self).add_edge(gateway, event)
                super(MyGraph, self).add_edge(gateway, target)
        for event in direct_succession[target]:
            if len(direct_succession[event]) == 1 and list(direct_succession[event])[0] == target:
                loop = True
                gateway = 'XORs {'+str(source)+', '+str(event)+'}->'+str(target) 
                self.add_xor_gateway(gateway, *args)
                super(MyGraph, self).add_edge(source, gateway)
                super(MyGraph, self).add_edge(event, gateway)
                super(MyGraph, self).add_edge(gateway, target)
        if not loop:
            super(MyGraph, self).add_edge(source, target)

    def add_and_split_gateway(self, source, targets, direct_succession, *args):
        gateway = 'ANDs '+str(source)+'->'+str(targets)        
        self.add_and_gateway(gateway,*args)
        self.add_source_edge(source, gateway, direct_succession, *args)
        for target in targets:
            super(MyGraph, self).add_edge(gateway, target)
 
    def add_xor_split_gateway(self, source, targets, direct_succession, *args):
        gateway = 'XORs '+str(source)+'->'+str(targets)
        self.add_xor_gateway(gateway, *args)
        self.add_source_edge(source, gateway, direct_succession, *args)
        for target in targets:
            super(MyGraph, self).add_edge(gateway, target)
 
    def add_and_merge_gateway(self, sources, target, direct_succession, *args):
        gateway = 'ANDm '+str(sources)+'->'+str(target)
        self.add_and_gateway(gateway,*args)
        super(MyGraph, self).add_edge(gateway,target)
        for source in sources:
            self.add_source_edge(source, gateway, direct_succession, *args)
 
    def add_xor_merge_gateway(self, sources, target, direct_succession, *args):
        gateway = 'XORm '+str(sources)+'->'+str(target)
        self.add_xor_gateway(gateway, *args)
        super(MyGraph, self).add_edge(gateway,target)
        for source in sources:
            self.add_source_edge(source, gateway, direct_succession, *args)

    def add_xor_to_and_gateway(self, sources, targets, direct_succession, *args):
        xor_gateway = 'XORm '+str(sources)
        and_gateway = 'ANDs ->'+str(targets)
        self.add_xor_gateway(xor_gateway, *args)
        self.add_and_gateway(and_gateway,*args)
        for source in sources:
            self.add_source_edge(source, xor_gateway, direct_succession, *args)
        for target in targets:
            super(MyGraph, self).add_edge(and_gateway, target)
        super(MyGraph, self).add_edge(xor_gateway, and_gateway)

    def add_xor_to_xor_gateway(self, source, skipped, target, direct_succession, *args):
        xor_gateway = 'XORs '+str(source)+'->'+str(skipped)
        xor_gateway_2 = 'XORm '+str(skipped)+'->'+str(target)
        self.add_xor_gateway(xor_gateway, *args)
        self.add_xor_gateway(xor_gateway_2,*args)
        self.add_source_edge(source, xor_gateway, direct_succession, *args)
        super(MyGraph, self).add_edge(xor_gateway, skipped)
        super(MyGraph, self).add_edge(skipped, xor_gateway_2)
        super(MyGraph, self).add_edge(xor_gateway, xor_gateway_2)
        super(MyGraph, self).add_edge(xor_gateway_2, target)

    def get_start_set_events(self, direct_succession):
        start_set_events = set()
        for key in direct_succession.keys():
            in_events = False
            for ev_cause in direct_succession.keys():
                if key in direct_succession.get(ev_cause, set()):
                    in_events = True
            if not in_events:
                start_set_events.add(key)
        return start_set_events

    def get_start_set_events_from_trace(self, trace):
        start_set_events = set()
        for t in trace:
            start_set_events.add(t[0])
        return start_set_events

    def get_end_set_events_from_trace(self, trace):
        end_set_events = set()
        for t in trace:
            end_set_events.add(t[len(t)-1])
        return end_set_events

    def get_end_set_events(self, direct_succession, parallel_events):
        end_set_events = set()
        for ev_cause, events in direct_succession.items():
            if len(events) == 0:
                end_set_events.add(ev_cause)
                continue
            parallel_end = True
            for event in events:
                if tuple((ev_cause, event)) not in parallel_events or ev_cause in direct_succession[event]:
                    parallel_end = False
            if parallel_end:
                end_set_events.add(ev_cause)
        return end_set_events

    def get_causality(self, direct_succession) -> Dict[str, Set[str]]:
        causality = defaultdict(set)
        for ev_cause, events in direct_succession.items():
            for event in events:
                if ev_cause not in direct_succession.get(event, set()):
                    causality[ev_cause].add(event)
        return dict(causality)

    def get_inv_causality(self, causality) -> Dict[str, Set[str]]:
        inv_causality = defaultdict(set)
        for key, values in causality.items():
            for value in values: 
              inv_causality[value].add(key)
        return {k: v for k, v in inv_causality.items() if len(v) > 1}

    def get_parallel_events(self, direct_succession):
        parallel_events = set()
        for ev_cause, events in direct_succession.items():
            for event in events:
                if ev_cause in direct_succession.get(event, set()):
                    parallel_events.add((ev_cause, event))
        return parallel_events

    def get_edges(self, direct_succession, parallel_events, causality, inv_causality, end_set_events):
        edges = set()
        for ev_cause, events in causality.items():
            if len(events) == 1:  
              for event in events:
                if tuple((ev_cause, event)) not in parallel_events and event not in inv_causality.keys() and ev_cause not in end_set_events:
                    edges.add((ev_cause, event))
        return edges

    def print_relations(self, direct_succession, start_set_events, causality, inv_causality, parallel_events, end_set_events, edges):
      print("Direct succession:")
      print(direct_succession)

      print("Start set events:")
      print(start_set_events)

      print("Causality:")
      print(causality)

      print("Inverse causality:")
      print(inv_causality)

      print("Parallel events:")
      print(parallel_events)

      print("End set events:")
      print(end_set_events)

      print("Edges:")
      print(edges)

      return

    def create_and_display_graph(self, model_name, filename="", succession=None):
        if succession == None and filename != "" and filename.endswith(".csv"):
            trace = self.get_trace_from_csv(filename)
            events = self.get_events_from_csv(filename)
            direct_succession = self.get_direct_succession(trace, events)
        elif succession == None and filename != "" and filename.endswith(".xes"):
            trace = self.get_trace_from_xes(filename)
            events = self.get_events_from_xes(filename)
            direct_succession = self.get_direct_succession(trace, events)
        elif succession != None and filename == "":
            direct_succession = succession
        else:
            print("Provide direct succession or csv/xes file!")
            return

        print(trace)
        print(events)

        # getting start set events
        start_set_events = self.get_start_set_events(direct_succession)

        # getting parallel events
        parallel_events = self.get_parallel_events(direct_succession)

        # getting end set events
        end_set_events = self.get_end_set_events(direct_succession, parallel_events)

        if len(start_set_events) == 0 and filename != "":
            start_set_events = self.get_start_set_events_from_trace(trace)
          
        if len(end_set_events) == 0 and filename != "":
            end_set_events = self.get_end_set_events_from_trace(trace)

        # getting causality
        causality = self.get_causality(direct_succession)

        # getting inverse causality
        inv_causality = self.get_inv_causality(causality)

        # getting edges
        edges = self.get_edges(direct_succession, parallel_events, causality, inv_causality, end_set_events)

        # print relations
        self.print_relations(direct_succession, start_set_events, causality, inv_causality, parallel_events, end_set_events, edges)

        # adding start event
        self.add_event("start")
        if len(start_set_events) > 1:
            if tuple(start_set_events) in parallel_events: 
                self.add_and_split_gateway("start",start_set_events, direct_succession)
            else:
                self.add_xor_split_gateway("start",start_set_events, direct_succession)
        else: 
            self.add_start_edge("start",list(start_set_events)[0], direct_succession, start_set_events)

        causality_blacklist = []
        inv_causality_blacklist = []
        end_set_events_blacklist = []

        for event in causality:
            if event in causality_blacklist:
                continue
            if len(causality[event]) > 1:
                if tuple(causality[event]) in parallel_events:
                    sources = []
                    targets = []
                    for event2 in causality:
                        if causality[event] == causality[event2] and event != event2:
                            causality_blacklist.append(event)
                            causality_blacklist.append(event2)
                            sources.append(event)
                            sources.append(event2)
                            for ev in causality[event]:
                                inv_causality_blacklist.append(ev)
                                targets.append(ev)
                    if len(sources) > 0 and len(targets) > 0:
                        self.add_xor_to_and_gateway(sources, targets, direct_succession)
                elif event in end_set_events:
                    causality_blacklist.append(event)
                    for ev in causality[event]:
                        if ev in inv_causality and event in inv_causality[ev]:
                            inv_causality[ev].remove(event)
                else:
                    for ev in causality[event]:
                        for ev2 in causality[event]:
                            if ev2 in causality.get(ev, set()):
                                causality_blacklist.append(event)
                                causality_blacklist.append(ev)
                                inv_causality_blacklist.append(ev2)
                                self.add_xor_to_xor_gateway(event, ev, ev2, direct_succession)
                              

        # adding split gateways based on causality
        for event in causality:
            if event in causality_blacklist:
                continue
            if len(causality[event]) > 1:
                if len(causality[event]) > 2:
                    parallel = True
                    for ev in causality[event]:
                        for ev2 in causality[event]:
                            if tuple((ev, ev2)) not in parallel_events and ev != ev2:
                                parallel = False
                    if parallel:
                        self.add_and_split_gateway(event, causality[event], direct_succession)
                    else:
                        self.add_xor_split_gateway(event,causality[event], direct_succession)
                elif tuple(causality[event]) in parallel_events:
                    self.add_and_split_gateway(event,causality[event], direct_succession)
                else:
                    self.add_xor_split_gateway(event,causality[event], direct_succession)
        
        # adding merge gateways based on inverted causality
        for event in inv_causality:
            if event in inv_causality_blacklist:
                continue
            if len(inv_causality[event]) > 1:
                if len(inv_causality[event]) > 2:
                    parallel = True
                    for ev in inv_causality[event]:
                        for ev2 in inv_causality[event]:
                            if tuple((ev, ev2)) not in parallel_events and ev != ev2:
                                parallel = False
                    if parallel:
                        self.add_and_merge_gateway(inv_causality[event],event, direct_succession)
                    else:
                        self.add_xor_merge_gateway(inv_causality[event],event, direct_succession)
                elif tuple(inv_causality[event]) in parallel_events:
                    self.add_and_merge_gateway(inv_causality[event],event, direct_succession)
                else:
                    self.add_xor_merge_gateway(inv_causality[event],event, direct_succession)
            elif len(inv_causality[event]) == 1:
                source = list(inv_causality[event])[0]
                self.add_edge(source,event)

        # adding rest of edges
        for edge in edges:
          self.add_target_edge(edge[0], edge[1], direct_succession)

        # adding end event
        self.add_end_event("end")
        if len(end_set_events) > 1:
            if tuple(end_set_events) in parallel_events: 
                self.add_and_merge_gateway(end_set_events,"end", direct_succession)
            else:
                self.add_xor_merge_gateway(end_set_events,"end", direct_succession)    
        else:
            event = list(end_set_events)[0]
            if event in causality:
                targets = causality[event]
                targets.add("end")
                self.add_xor_split_gateway(event, targets, direct_succession)
            else:    
                self.add_edge(list(event)[0],"end")

        self.draw(model_name, prog='dot')
        display(Image(model_name))