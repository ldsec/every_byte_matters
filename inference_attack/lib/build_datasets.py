import csv
import glob
import math
import os
import pickle
import sys
from functools import reduce
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
import lib.pretty_print as pp

SOURCES_FILE_IGNORE = ["Initial Tests", ".cache", ".sync", ".vscode", "lib", "share", "bin", "etc", "include", ".ipynb_checkpoints"]
MAX_CAPTURE_DURATION = 30

class bcolors:
    FAIL = '\033[91m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

# --------------------------------------
# Dataset Parsing

# Finds CSV datasets recursively
def find_sources(folder='data/'):
    global SOURCES_FILE_IGNORE

    sources_files = []
    
    files = glob.glob(folder+'*.csv', recursive=True)
    for file in files:
        ignore = False
        for ignore_pattern in SOURCES_FILE_IGNORE:
            if ignore_pattern in file:
                ignore = True
        if not ignore:
            sources_files.append(file.replace('./', ''))

    return sorted(sources_files)

# Parses a CSV dataset
all_columns = set()
def extract_columns(dataset_file):
    global all_columns

    with open(dataset_file, "r") as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        for i, line in enumerate(csv_reader):
            all_columns.update(line)
            return

def find_common_columns(sources_files):
    global all_columns
    
    all_columns = set()
    for source in sources_files:
        extract_columns(source)
    all_columns = list(all_columns)

packet_store = dict() # dataset_file => [packetID => layers]
ID_COLUMN = "Packet #"

# creates a new packet with all columns (from the union of each dataset's columns)
def new_packet():
    global all_columns

    packet = dict()
    for column in all_columns:
        packet[column] = ""
    
    return packet

def parse_csv(csv_file):
    """
    Reads a .csv file and puts its raw contents in packet_store[csv_file]
    """
    global packet_store, all_columns

    dataset_packet_store = dict() # packetID => packets
    headers = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')

        for i, line in enumerate(reader):
            if i == 0:
                headers = line
                continue
                
            packet = new_packet()

            for j, item in enumerate(line):
                key = headers[j]
                val = item

                if key not in packet:
                    print("Fatal: column", key, "not found in packet; all_columns is", all_columns)
                    sys.exit(1)
                packet[key] = val

            packet_id = toInt(packet[ID_COLUMN].replace('\'', ''))
            if packet_id not in dataset_packet_store:
                dataset_packet_store[packet_id] = []
            
            dataset_packet_store[packet_id].append(packet)

    packet_store[csv_file] = dataset_packet_store

# Cleanup
def toInt(s, default=0):
    if s.strip() == "":
        return default
    return int(s.replace('\'', ''))

def toFloat(s, default=0):
    try:
        if s.strip() == "":
            return default
        return float(s.replace('\'', ''))
    except:
        return default

def extract_payload_length(payload_string, default=0):
    payload_string = payload_string.strip()
    if payload_string == "" or "No data" in payload_string:
        return default
    parts = payload_string.split(' ')
    return toInt(parts[0])

def extract_payload_bytes(payload_string, default=[]): # format: "4 bytes (00 11 22 33)"
    payload_string = payload_string.strip()
    if payload_string == "" or "No data" in payload_string or not "(" in payload_string or not ")" in payload_string:
        return default
    cut1 = payload_string[payload_string.find("(")+1:]
    cut2 = cut1[:cut1.find(")")]
    return cut2

def packet_store_cleanup(dataset_packet_store): #dataset_packet_store is packetID => packets
    for packet_id in dataset_packet_store:
        layers = dataset_packet_store[packet_id]
        for layer in layers:
            layer[ID_COLUMN] = toInt(layer[ID_COLUMN])
            layer["Time"] = toFloat(layer["Time"], default=-1)
            layer["Time delta"] = toFloat(layer["Time delta"], default=-1)
            layer["PayloadLength"] = extract_payload_length(layer["Payload"], default=0)
            layer["PayloadRaw"] = extract_payload_bytes(layer["Payload"])

def packets_to_timesize_tuples(packets):
    xy = dict(xs=[], ys=[])
    
    packets_ids = list(packets.keys())
    packets_ids.sort(reverse=False)

    t0 = -1000

    for packet_id in packets_ids:
        for layer in packets[packet_id]:

            t = float(layer['Time'])
            if t0 == -1000:
                t0 = t
            y = int(layer['PayloadLength'])
            
            direction = 1
            if not "master" in layer['Transmitter'].lower():
                direction = -1

            if t-t0 <= MAX_CAPTURE_DURATION:
                xy['xs'].append(t-t0)
                xy['ys'].append(direction * y)
    return xy

# --------------------------------------
# Auxiliary functions (print & plot)

def pad(s, length, max_length = -1):
    s = str(s)
    if max_length > -1 and len(s) > max_length:
        s = s[:max_length-3]+"..."
    if len(s) == length:
        return s
    p = " " * (length - len(s))
    return s+p

def print_packet(p):
    isFirstLayer = True

    for layer in p:
        size = "0"
        if layer["PayloadLength"] != -1:
            size = str(layer["PayloadLength"])+"B ("+str(layer["PayloadRaw"])+")"

        details = ""
        if isFirstLayer:
            details = " ("+str(layer["Communication"])+")"
        print(str(layer[ID_COLUMN]) + ": T="+str(layer['Time']) + " P=" + size + ' Tx=' + layer['Originator'] + ' Item='+layer['Item'].strip() + details)

def plot(xs, ys, title="Title not set", bar_width=0.2):
    xs_nonzero = []
    ys_nonzero = []
    xs_zero = []
    ys_zero = []

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if y > 0:
            xs_nonzero.append(x)
            ys_nonzero.append(abs(y))
        else:
            xs_zero.append(x)
            ys_zero.append(-10) # special value to be visible on the graph

    if len(xs_zero)>0:
        plt.bar(xs_zero, ys_zero, color='r', align='center', linewidth=0, width=bar_width)
    if len(xs_nonzero)>0:
        plt.bar(xs_nonzero, ys_nonzero, color='b', align='center', linewidth=0, width=bar_width * 2)

    plt.xlabel('Time [s]')
    plt.ylabel('Packet size [B]')
    plt.title(title)

    legends = []
    if len(xs_zero)>0:
        legends.append('NULLs')
    if len(xs_nonzero)>0:
        legends.append('Data')

    plt.legend(legends,loc=2)
    
    plt.show()

# Cache
def cache(object_uri, object_to_cache):
    if not os.path.isdir('.cache'):
        os.mkdir('.cache')
    full_destination = ".cache/" + object_uri.replace('/', '_')
    try:
        os.remove(full_destination)
    except:
        pass
    with open(full_destination, "wb") as f:
        pickle.dump(object_to_cache, f)

def load_cache(object_uri):
    full_destination = ".cache/" + object_uri.replace('/', '_')
    if os.path.isfile(full_destination):
        #print("Loading", object_uri, "from cache (" + str(full_destination)+")")
        
        with open(full_destination, "rb") as f:
            return pickle.load(f)
    return None

def csv_to_xy_features(source_file):
    """
    Returns [(time,size)] for a given .csv source_file.
    Assumes packet_store has been filled (with parse_dataset) (not necessary if data is cached).
    """
    f = source_file+".xy"

    xy_features = load_cache(f)
    if xy_features == None:
        if not source_file in packet_store or packet_store[source_file] == None:
            print("Cache miss, loading and cleaning file", source_file)

            parse_csv(source_file)
            packet_store_cleanup(packet_store[source_file])

            cache(source_file, packet_store[source_file])

        xy_features = packets_to_timesize_tuples(packet_store[source_file])
        cache(f, xy_features)
    return xy_features

def csv_file_to_xy_events(source_file, timeout=4.5):
    """
    Returns [[(time,size)],] separated by events for a given .csv source_file.
    Assumes packet_store has been filled (with parse_dataset) (not necessary if data is cached).
    """
    f = source_file+"."+str(timeout)+".events.xy"

    xy_events = load_cache(f)
    if xy_events == None:
        xy_features = csv_to_xy_features(source_file)
        xy_events = cut_in_events(xy_features, timeout=timeout)
        cache(f, xy_events)
        
    return xy_events

# --------------------------------------
# Events

def cut_in_events(xy_features, timeout=4.5):
    """
    Returns [[(time, size)],] where each list is separated by at least timeout time, from a given [(time,size)]
    """
    events = []
    def trim_and_add_event(start, end):
        start_trim = start
        while ys[start_trim] == 0 :
            start_trim += 1
            if start_trim >= len(ys):
                return None
        end_trim = end
        while ys[end_trim] == 0 :
            end_trim -= 1

        if start_trim > end_trim:
            print("You broke the matrix", start, end, start_trim, end_trim, len(xs), len(ys))
            sys.exit(1)

        xs_copy = xs[start_trim:end_trim]
        ys_copy = ys[start_trim:end_trim]
        events.append(dict(xs=xs_copy, ys=ys_copy))
        

    xs = xy_features['xs']
    ys = xy_features['ys']

    if timeout == -1:
        return [dict(xs=xs, ys=ys)]

    feature_start = 0
    last_activity_index = -1  # not even one activity seen yet, never create an event in this case
    index = 0

    while index < len(xs):

        if last_activity_index != -1 and xs[index] - xs[last_activity_index] > timeout:
            trim_and_add_event(feature_start, index)

            feature_start = index
            last_activity_index = -1 # not even one activity seen yet

        if ys[index] > 0:
            last_activity_index = index
        index += 1

    # don't forget the last event
    if feature_start < len(xs)-1:
        trim_and_add_event(feature_start, len(xs)-1)

    events = [e for e in events if e is not None]
    return events

def cut_all_datasets_in_events(sources_files, folder='data/', timeout=-1):
    """
    Returns a map[device][app][action] => list[[(time, size)],] where list is cut by events
    Second return value is the same structure, but with # of packets instead of events
    """
    events = dict()
    counts = 0
    ignored = 0

    max_size = 0

    # cut each file in events
    for s in sources_files:
        fname = s.replace(folder, '')
        parts = fname.split("_")
        device, app, action = parts[0], parts[1], parts[2]

        if not device in events:
            events[device] = dict()
        if not app in events[device]:
            events[device][app] = dict()
        if not action in events[device][app]:
            events[device][app][action] = []
        
        file_events = csv_file_to_xy_events(s, timeout=timeout) # cut each trace into subtraces after 4.5 idle time
        counts += len(file_events)

        event_index = 0
        for event in file_events:
            n_packets = len(event['xs'])
            if n_packets == 0:
                print("Skipping", s, " (",device,app,action,") event",event_index, "as it has 0 packets")
                ignored += 1
                pass
            else:
                event['source'] = s + '_' + str(event_index)
                events[device][app][action].append(event)

            for e in event['ys']:
                max_size = max(max_size, abs(e))

            event_index += 1

    #print("Max size:", max_size)
    if ignored > 0:
        print("Note:", ignored, "samples ignored (0 packets)")

    return events, counts


# --------------------------------------
# General Dataset Operations

def split_test_train(sources_files, folder='data/', test_percentage=0.05):
    """
    Sorts input csv into map[device][app][action], then foreach action, split into test train
    (1 entry for test, remaining for train; typically that's 20% test 80% train)
    """
    sources_hierarchy = dict()

    for s in sources_files:
        fname = s.replace(folder, '')
        parts = fname.split("_")
        device, app, action = parts[0], parts[1], parts[2]

        if not device in sources_hierarchy:
            sources_hierarchy[device] = dict()
        if not app in sources_hierarchy[device]:
            sources_hierarchy[device][app] = dict()
        if not action in sources_hierarchy[device][app]:
            sources_hierarchy[device][app][action] = []

        sources_hierarchy[device][app][action].append(s)

    sources_test = []
    sources_train = []

    for device in sources_hierarchy:
        for app in sources_hierarchy[device]:
            for action in sources_hierarchy[device][app]:
                total = len(sources_hierarchy[device][app][action])
                n_test = round(total * test_percentage)
                current_n_test = 0

                for s in sorted(sources_hierarchy[device][app][action]):
                    if current_n_test<n_test:
                        sources_test.append(s)
                        current_n_test += 1
                    else:
                        sources_train.append(s)
                    
    return sources_test, sources_train

def split_classic_ble(sources_files, folder='data/'):
    """
    Sorts input csv into map[device][app][action]
    """
    sources_hierarchy = dict()

    def is_ble(s):
        fname = s.replace(folder, '')
        parts = fname.split("_")
        return parts[3].lower() == "ble"


    for s in sources_files:
        fname = s.replace(folder, '')
        parts = fname.split("_")
        device, app, action = parts[0], parts[1], parts[2]

        if not device in sources_hierarchy:
            sources_hierarchy[device] = dict()
        if not app in sources_hierarchy[device]:
            sources_hierarchy[device][app] = dict()
        if not action in sources_hierarchy[device][app]:
            sources_hierarchy[device][app][action] = []

        sources_hierarchy[device][app][action].append(s)

    sources_classic = []
    sources_ble = []

    for device in sources_hierarchy:
        for app in sources_hierarchy[device]:
            for action in sources_hierarchy[device][app]:
                for s in sorted(sources_hierarchy[device][app][action]):
                    if is_ble(s):
                        sources_ble.append(s)
                    else:
                        sources_classic.append(s)
                    
    return sources_classic, sources_ble


def split_test_train_non_mixed(sources_files, folder='data/', test_percentage=0.05):
    """
    Sorts input csv into map[device][app][action], then foreach action, split into test train
    (1 entry for test, remaining for train; typically that's 20% test 80% train)
    """
    sources_hierarchy = dict()

    def is_ble(s):
        fname = s.replace(folder, '')
        parts = fname.split("_")
        return parts[3].lower() == "ble"

    for s in sources_files:
        fname = s.replace(folder, '')
        parts = fname.split("_")
        device, app, action = parts[0], parts[1], parts[2]

        if not device in sources_hierarchy:
            sources_hierarchy[device] = dict()
        if not app in sources_hierarchy[device]:
            sources_hierarchy[device][app] = dict()
        if not action in sources_hierarchy[device][app]:
            sources_hierarchy[device][app][action] = []

        sources_hierarchy[device][app][action].append(s)

    sources_test_classic = []
    sources_test_ble = []
    sources_train_classic = []
    sources_train_ble = []

    for device in sources_hierarchy:
        for app in sources_hierarchy[device]:
            for action in sources_hierarchy[device][app]:

                total = len(sources_hierarchy[device][app][action])
                n_test = round(total * test_percentage)
                current_n_test = 0
                for s in sorted(sources_hierarchy[device][app][action]):
                    if current_n_test<n_test:
                        if is_ble(s):
                            sources_test_ble.append(s)
                        else:
                            sources_test_classic.append(s)
                        current_n_test += 1
                    else:
                        if is_ble(s):
                            sources_train_ble.append(s)
                        else:
                            sources_train_classic.append(s)
                    
    return sources_test_classic, sources_test_ble, sources_train_classic, sources_train_ble

def count_all_packets(events_all_devices):
    n_packets = dict()
    for device in events_all_devices:
        events = events_all_devices[device]
        if not device in n_packets:
            n_packets[device] = 0
        n = reduce(lambda a,b: a+b, [len(e['xs']) for e in events], 0) # just a deep sum
        n_packets[device] += n

    return n_packets

def equilibrate_events_across_app_or_action(events):
    counts = dict()
    for device in events:        
        for app in events[device]:

            if not app in counts:
                counts[app] = 0

            for action in events[device][app]:
                counts[app] += len(events[device][app][action])
    
    min_number_events = min(counts.values())

    print("Min number of events:", min_number_events)
    
    events_out = dict()

    # remove everything above the min # across devices
    for device in events:
        
        current_leaf_index = 0
        current_count = 0
        while current_count < min_number_events:
            for app in events[device]:
                for action in events[device][app]:
                    if len(events[device][app][action]) > current_leaf_index:                        
                        if not device in events_out:
                            events_out[device] = dict()
                        if not app in events_out[device]:
                            events_out[device][app] = dict()
                        if not action in events_out[device][app]:
                            events_out[device][app][action] = []
                        events_out[device][app][action].append(events[device][app][action][current_leaf_index])
                        current_count += 1
            current_leaf_index += 1
    return events_out


def equilibrate_events_across_action(events):
    counts = dict()
    for device in events:        
        for app in events[device]:

            if not app in counts:
                counts[app] = 0

            for action in events[device][app]:
                counts[app] += len(events[device][app][action])
    
    min_number_events = min(counts.values())

    print("Min number of events:", min_number_events)
    
    events_out = dict()

    # remove everything above the min # across devices
    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                if not device in events_out:
                    events_out[device] = dict()
                if not app in events_out[device]:
                    events_out[device][app] = dict()
                if not action in events_out[device][app]:
                    events_out[device][app][action] = []
                events_out[device][app][action].extend(events[device][app][action][:min_number_events])

    return events_out


def roughly_equilibrate_events_across_action(events, max_number_of_events=40):

    events_out = dict()

    # remove everything above the min # across devices
    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                if not device in events_out:
                    events_out[device] = dict()
                if not app in events_out[device]:
                    events_out[device][app] = dict()
                if not action in events_out[device][app]:
                    events_out[device][app][action] = []
                i = 0
                while i<len(events[device][app][action]) and i<max_number_of_events:
                    events_out[device][app][action].append(events[device][app][action][i])
                    i += 1
    return events_out

def equilibrate_events_across_chipset(events, CHIPSET_MAP):

    def getChipset(device):
        if not device in CHIPSET_MAP:
            print("Device", device, "not in chipset map")
            sys.exit(1)
        return CHIPSET_MAP[device]

    counts = dict()
    for device in events:
        chipset = getChipset(device)
        if not chipset in counts:
            counts[chipset] = 0
        
        for app in events[device]:
            for action in events[device][app]:
                counts[chipset] += len(events[device][app][action])
                #print(chipset, device, app, action, len(events[device][app][action]))
    
    print("Chipset event count pre-equalization:")
    pp.new_table()
    accu = 0
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
    pp.table_push("total:", accu)
    pp.table_print()

    min_number_events = min(counts.values())
    
    events_out = dict()

    events_by_chipset=dict()
    for device in events:
        chipset = getChipset(device)

        if not chipset in events_by_chipset:
            events_by_chipset[chipset] = dict()
        
        events_by_chipset[chipset][device] = events[device]


    # remove everything above the min # across devices
    for chipset in events_by_chipset:
        
        current_leaf_index = 0
        current_count = 0
        while current_count < min_number_events:
            for device in events_by_chipset[chipset]:
                for app in events_by_chipset[chipset][device]:
                    for action in events_by_chipset[chipset][device][app]:
                        if len(events_by_chipset[chipset][device][app][action]) > current_leaf_index:                        
                            if not chipset in events_out:
                                events_out[chipset] = dict()                        
                            if not device in events_out[chipset]:
                                events_out[chipset][device] = dict()
                            if not app in events_out[chipset][device]:
                                events_out[chipset][device][app] = dict()
                            if not action in events_out[chipset][device][app]:
                                events_out[chipset][device][app][action] = []
                            events_out[chipset][device][app][action].append(events_by_chipset[chipset][device][app][action][current_leaf_index])
                            current_count += 1
            current_leaf_index += 1

    counts = dict()
    for chipset in events_out:
        if not chipset in counts:
            counts[chipset] = 0
        
        for device in events_out[chipset]:
            for app in events_out[chipset][device]:
                for action in events_out[chipset][device][app]:
                    counts[chipset] += len(events_out[chipset][device][app][action])
                    #print(chipset, device, app, action, len(events_out[chipset][device][app][action]))
    
    print("Chipset event count post-equalization:")
    pp.new_table()
    accu = 0
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
    pp.table_push("total:", accu)
    pp.table_print()

    return events_out


def equilibrate_events_across_devices(events):
    counts = dict()
    for device in events:
        if not device in counts:
            counts[device] = 0
        
        for app in events[device]:
            for action in events[device][app]:
                counts[device] += len(events[device][app][action])
                #print(device, app, action, len(events[device][app][action]))
    
    print("Devices event count pre-equalization:")
    pp.new_table()
    accu = 0
    minmax = [0, 0]
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
        minmax[0] = min(counts[k], minmax[1])
        minmax[1] = max(counts[k], minmax[1])
    pp.table_push("total:", accu, "min", minmax[0], "max", minmax[1])
    pp.table_print()

    min_number_events = min(counts.values())
    
    events_out = dict()

    # remove everything above the min # across devices
    for device in events:
        
        current_leaf_index = 0
        current_count = 0
        while current_count < min_number_events:
            for app in events[device]:
                for action in events[device][app]:
                    if len(events[device][app][action]) > current_leaf_index:                        
                        if not device in events_out:
                            events_out[device] = dict()
                        if not app in events_out[device]:
                            events_out[device][app] = dict()
                        if not action in events_out[device][app]:
                            events_out[device][app][action] = []
                        events_out[device][app][action].append(events[device][app][action][current_leaf_index])
                        current_count += 1
            current_leaf_index += 1

    counts = dict()
    for device in events_out:
        if not device in counts:
            counts[device] = 0
        
        for app in events_out[device]:
            for action in events_out[device][app]:
                counts[device] += len(events_out[device][app][action])
                #print(device, app, action, len(events_out[device][app][action]))
    
    print("Devices event count post-equalization:")
    pp.new_table()
    accu = 0
    minmax = [0, 0]
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
        minmax[0] = min(counts[k], minmax[1])
        minmax[1] = max(counts[k], minmax[1])
    pp.table_push("total:", accu, "min", minmax[0], "max", minmax[1])
    pp.table_print()

    return events_out

def equilibrate_events_across_apps(events):
    counts = dict()
    for device in events:        
        for app in events[device]:
            if not app in counts:
                counts[app] = 0
            for action in events[device][app]:
                counts[app] += len(events[device][app][action])
    
    print("Devices app count pre-equalization:")
    pp.new_table()
    accu = 0
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
    pp.table_push("total:", accu)
    pp.table_print()

    if len(counts.values())== 0:
        return events

    min_number_events = min(counts.values())
    
    events_out = dict()

    # remove everything above the min # across devices
    for device in events:
        for app in events[device]:
            current_leaf_index = 0
            current_count = 0
            while current_count < min_number_events:
                for action in events[device][app]:
                    if len(events[device][app][action]) > current_leaf_index:                        
                        if not device in events_out:
                            events_out[device] = dict()
                        if not app in events_out[device]:
                            events_out[device][app] = dict()
                        if not action in events_out[device][app]:
                            events_out[device][app][action] = []
                        events_out[device][app][action].append(events[device][app][action][current_leaf_index])
                        current_count += 1
            current_leaf_index += 1

    counts = dict()
    for device in events_out:
        for app in events_out[device]:
            if not app in counts:
                counts[app] = 0
            for action in events_out[device][app]:
                counts[app] += len(events_out[device][app][action])
    
    print("Devices event count post-equalization:")
    pp.new_table()
    accu = 0
    for k in counts:
        pp.table_push(k, counts[k])
        accu += counts[k]
    pp.table_push("total:", accu)
    pp.table_print()

    return events_out

# --------------------------------------
# Main logic of this script:

def rebuild_all_datasets(sources_files=None, force_rebuild=True):
    global all_columns, packet_store

    if sources_files is None:
        sources_files = find_sources()
    find_common_columns(sources_files)
    
    #print("Common columns:", all_columns)

    return

    sources_files = sorted(sources_files)

    for source_file in sources_files:
        packet_store[source_file] = load_cache(source_file)

        if force_rebuild or packet_store[source_file] == None:
            print("Loading and cleaning dataset", source_file)

            parse_csv(source_file)
            packet_store_cleanup(packet_store[source_file])

            cache(source_file, packet_store[source_file])
        else:
            print("Loading in memory", source_file)
        
        csv_to_xy_features(source_file) # map but more importantly cache the results
        
    for source_file in sources_files:
        print("Dataset", source_file, "contains", len(packet_store[source_file]), "packets")

if __name__ == "__main__":
    print("Parsing all files in current directory")
    rebuild_all_datasets()
