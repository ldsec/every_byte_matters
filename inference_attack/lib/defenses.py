import math
from math import floor 
from math import ceil 
import random
import sys
import numpy as np
from operator import itemgetter

def flat_perc(a, b):
    if a == 0:
        return str(round(b-a, 0)) + "B = INF%"
    return str(round(b-a, 0)) + "B = " + str(round(100*b/a, 2))+"%"

def array_stats(data):
    if len(data) == 0:
        return "NaN"
    return round(np.mean(data), 2)

def array_stats_str(data):
    return str(array_stats(data))

class CostAggregate:
    traces = dict()
    def __init__(self):
        self.traces = dict()

    def add_cost(self, c, label=None):
        #print("Cost", c)
        if not label in self.traces:
            self.traces[label] = []
            
        self.traces[label].append(c)


    def to_serialized_array(self):
        o = dict()
        for label in self.traces:
            o[label] = []
            for c in self.traces[label]:
                o[label].append(c.packets)
        return o

    def from_serialized_array(self, d):
        self.traces = dict()
        for label in d:
            self.traces[label] = []
            for cost_data in d[label]:
                cost = Cost()
                cost.packets = cost_data
                self.traces[label].append(cost)
        return self

    def stats_for_label(self, label=None):
        data = []

        for l in self.traces:
            if label is None or l==label:
                for t in self.traces[l]:
                    sum_before, sum_after = t.overall_overhead()
                    sum_dummies = t.sum_dummies()
                    sum_pad = t.sum_pad()
                    oh_ind = t.individual_overheads()
                    dur_flat = t.duration()
                    dur_ind = t.individual_delays()

                    if sum_before==0:
                        continue


                    data.append([sum_after, sum_before, sum_dummies, sum_pad, array_stats(oh_ind), dur_flat, array_stats(dur_ind)])

        # mean per column
        data = np.array(data)
        data = np.transpose(data)
        out = []
        for array in data:
            out.append(array_stats(array))

        out = np.transpose(out)
        return out

    def stats(self):
        return self.stats_for_label()

    def detailed_stats(self):
        o = dict()
        for label in self.traces:
            o[label] = self.stats_for_label(label=label)
        return o

    def __str__(self):
        out = self.stats()
        if len(out) == 0:
            return "[No cost]"
        [sum_after, sum_before, sum_dummies, sum_pad, oh_ind, dur_flat, dur_ind] = out
        return "[Total: {}; dummies {}, pad {}, individual mean overhead {}B, extra duration {}s, individual mean delay {}s]".format(
            flat_perc(sum_before, sum_after), flat_perc(sum_before, sum_before+sum_dummies), flat_perc(sum_before, sum_before+sum_pad), oh_ind, dur_flat, dur_ind)


class Cost:
    # (Origin, Padded), each being None or [time, size]
    packets = []
    def __init__(self):
        self.packets = []

    def new_dummy(self, time, size):
        self.packets.append([None, [time, size]])

    def new_packet(self, t1, y1, t2, y2):
        #print("New packet", [[t1, y1], [t2, y2]])
        self.packets.append([[t1, y1], [t2, y2]])
    
    def overall_overhead(self):
        sum_before = 0
        sum_after = 0
        for p in self.packets:
            before, after = p
            if before is not None:
                sum_before += abs(before[1])
            sum_after += abs(after[1])

        return sum_before, sum_after

    def sum_dummies(self):
        _sum = 0
        for p in self.packets:
            before, after = p
            if before is None:
                _sum += abs(after[1])
        return _sum

    def sum_pad(self):
        _sum = 0
        for p in self.packets:
            before, after = p
            if before is not None:
                _sum += abs(after[1]-before[1])
        return _sum
    
    def individual_overheads(self):
        oh = []
        for p in self.packets:
            before, after = p
            if before is None:
                continue
            oh.append(abs(after[1]-before[1]))
        return oh
    
    def individual_delays(self):
        oh = []
        for p in self.packets:
            before, after = p
            if before is None:
                continue
            oh.append(after[0]-before[0])
        return oh

    def duration(self):
        self.packets.sort(key=lambda p: p[1][0])
        
        last_real = len(self.packets)-1
        while last_real >= 0:
            before, _ = self.packets[last_real]
            if before is not None:
                break
            last_real-=1

        if last_real == -1:
            return 0

        if last_real == len(self.packets)-1:
            return self.packets[last_real][1][0] - self.packets[last_real][0][0]


        # last noised/dummy's timestamp - last real's timestamp
        return self.packets[-1][1][0] - self.packets[last_real][0][0]
    
    def __str__(self):
        self.packets.sort(key=lambda p: p[1][0])
        sum_before, sum_after = self.overall_overhead()
        sum_dummies = self.sum_dummies()
        sum_pad = self.sum_pad()

        n_dummies = len([x for x in self.packets if x[0] is None])
        return "total: {}; {} dummies: {}; {} padded packets: {}, individual mean overhead {}B, extra duration {}s, individual mean delay {}s".format(
            flat_perc(sum_before, sum_after), n_dummies, flat_perc(sum_before, sum_before+sum_dummies), len(self.packets)-n_dummies, flat_perc(sum_before, sum_before+sum_pad), array_stats(self.individual_overheads()), round(self.duration(), 2), array_stats(self.individual_delays()))

def pad_cla(xs, ys):
    return pad(xs, ys, 1000)

def pad_ble(xs, ys):
    return pad(xs, ys, 255)

def pad(xs, ys, padded_size=128):
    ys2 = [0] * len(ys)
    cost = Cost()

    for i in range(len(ys)):
        if ys[i] == 0:
            continue #skip control packets

        y2 = pad_to_multiple(ys[i], padded_size)
        ys2[i] = y2
        cost.new_packet(xs[i], ys[i], xs[i], y2)

    return xs, ys2, cost

def pad_to_multiple(y, packet_size=128):
    if y % packet_size == 0:
        return y
    if y < 0:
        return y - y%packet_size
    else:
        return y + (packet_size - y%packet_size)


def delay_nextunit(xs, ys, precision=1.0):
    shifted_packets = dict()

    # delay
    for i in range(len(xs)):
        x2 = math.ceil(xs[i]*precision)/precision # round up

        if not x2 in shifted_packets:
            shifted_packets[x2] = []
        shifted_packets[x2].append(i)

    
    times = sorted(list(shifted_packets.keys()))

    xs2 = []
    ys2 = []
    cost = Cost()

    for t in times:
        _sum = 0
        for index in shifted_packets[t]:
            _sum += abs(ys[index])

        if _sum == 0:
            xs2.append(t)
            ys2.append(0)
            continue #skip control packets

        xs2.append(t)
        ys2.append(_sum) #combine all packets

        # record cost per packet
        i = 0
        while i < len(shifted_packets[t]):
            index = shifted_packets[t][i]
            cost.new_packet(xs[index], ys[index], t, ys[index])
            i += 1
            
    return xs2, ys2, cost

def delay_nextunit_then_pad_cla(xs, ys):
    return delay_nextunit_then_pad(xs, ys, padded_size=1000)

def delay_nextunit_then_pad_ble(xs, ys):
    return delay_nextunit_then_pad(xs, ys, padded_size=255)

def delay_nextunit_then_pad(xs, ys, precision=1.0, padded_size=128):
    shifted_packets = dict()

    # delay
    for i in range(len(xs)):
        x2 = math.ceil(xs[i]*precision)/precision # round up

        if not x2 in shifted_packets:
            shifted_packets[x2] = []
        shifted_packets[x2].append(i)

    
    times = sorted(list(shifted_packets.keys()))

    xs2 = []
    ys2 = []
    cost = Cost()

    for t in times:
        _sum = 0
        for index in shifted_packets[t]:
            _sum += abs(ys[index])

        if _sum == 0:
            xs2.append(t)
            ys2.append(0)
            continue #skip control packets


        xs2.append(t)
        ys2.append(pad_to_multiple(_sum, padded_size)) #combine all packets in a padded version

        # record cost per packet
        i = 0
        while i < len(shifted_packets[t]):
            index = shifted_packets[t][i]

            if i != len(shifted_packets)-1:
                cost.new_packet(xs[index], ys[index], t, ys[index])
            else:
                # only "last" packet of the sequence is padded so the sequence has the correct size
                cost.new_packet(xs[index], ys[index], t, ys[index]+(padded_size-_sum))
            i += 1
            
    return xs2, ys2, cost

def front_cla(xs, ys):
    return front(xs, ys)

def front_ble(xs, ys):
    return front(xs, ys)

def normal_cla(xs, ys):
    return front(xs, ys, distrib='normal')

def normal_ble(xs, ys):
    return front(xs, ys, distrib='normal')

# min_W, max_W, min_dummies, max_dummies
front_params = [6, 7, 300, 301]

def front(xs, ys, distrib='rayleigh'):
    global front_params

    cost = Cost()

    # for a dummy at time t, merge with following packet if separated by less than DELTAX sec
    MERGE_DELTAX = 0.01

    [min_W, max_W, min_dummies, max_dummies] = front_params
    
    def gen_dummies_times(t_shift=0, t_cutoff=-1):
        #W = np.random.uniform(min_W, max_W)
        W = np.random.randint(min_W, max_W)
        n_dummies = np.random.randint(min_dummies, max_dummies)
        
        dummies_times = []
        if distrib=='rayleigh':
            dummies_times = sorted(np.random.rayleigh(W, n_dummies))
        elif distrib=='normal':
            dummies_times = sorted(abs(np.random.normal(W, 10, n_dummies)))

        dummies_times = [t+t_shift for t in dummies_times]
        dummies_times = [t for t in dummies_times if t<t_cutoff]

        return dummies_times

    # alters "trace" and also returns new packets to be appended to trace
    def inject_or_merge_dummies(trace, dummies_t, direction=1):
        dummies_out = []
        cur_i = 0
        for dummy_time in dummies_t:
            while cur_i < len(xs) and (xs[cur_i]< dummy_time or ys[cur_i] * direction <= 0):
                cur_i += 1
            
            dummy_size = sample_size()
            if cur_i < len(trace) and ys[cur_i] * direction > 0 and xs[cur_i] - dummy_time < MERGE_DELTAX:
                trace[cur_i][1] += direction * dummy_size
                cost.new_dummy(trace[cur_i][0], dummy_size)
            else:
                dummies_out.append([dummy_time, direction * dummy_size])
                cost.new_dummy(dummy_time, dummy_size)
        return dummies_out

    zipped = list(zip(xs, ys))
    for x,y in zipped:
        cost.new_packet(x, y, x, y) # no overhead
    noisy_trace = np.array([x for x in zipped]) # copy

    first_incoming_time = 0
    for time, size in noisy_trace:
        if size < 0:
            first_incoming_time = time
            break
    
    last_pkt_time = noisy_trace[-1][0]

    client_dummies_t = gen_dummies_times(t_shift=0, t_cutoff=last_pkt_time)

    client_dummies = inject_or_merge_dummies(noisy_trace, client_dummies_t, direction=1)

    server_dummies_t = gen_dummies_times(t_shift=first_incoming_time)
    server_dummies = inject_or_merge_dummies(noisy_trace, server_dummies_t, direction=-1)

    if len(client_dummies) > 0:
        noisy_trace = np.concatenate((noisy_trace, client_dummies), axis=0)
    if len(server_dummies) > 0:
        noisy_trace = np.concatenate((noisy_trace, server_dummies), axis=0)
    
    np.sort(noisy_trace, axis=0)
    xs = noisy_trace[:, 0].tolist()
    ys = noisy_trace[:, 1].tolist()

    return xs, ys, cost



def apply_defense(event, defense=None):
    
    flavor = None
    if "_Classic_" in event['source']:
        flavor = 'CLA'
    if "_BLE_" in event['source']:
        flavor = 'BLE'
    if flavor is None:
        print("Can't identify Bluetooth variant")
        sys.exit(1)

    #print("Applying", defense, flavor, "to", event['source'])

    if defense == None:
        return event, Cost()

    elif defense in defenses:
        fn = defenses[defense][flavor]
        xs, ys, cost = fn(event['xs'], event['ys'])
        event_defended = dict(xs=xs, ys=ys)

        #print("cost", cost)

        return event_defended, cost


size_distribs = []

def build_size_distribution(events):
    global size_distribs

    total = 0
    sizes_hist = dict()
    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                for sample in events[device][app][action]:
                    for s in [y for y in sample['ys'] if y != 0]:
                        if not s in sizes_hist:
                            sizes_hist[s] = 0
                        sizes_hist[s] += 1
                        total += 1

    #print("Size distribution computed", sizes_hist)

    size_distribs = [0] * total
    i = 0
    for k in sorted(sizes_hist.keys()):
        while sizes_hist[k] > 0:
            size_distribs[i] = k
            sizes_hist[k] -= 1

            i += 1

def sample_size():
    global size_distribs

    if len(size_distribs) == 0:
        print("Can't sample from distribution, it hasn't been computed")
        sys.exit(1)
    
    return random.choice(size_distribs)

defenses = dict() 
defenses['pad'] = dict()
defenses['pad']['CLA'] = pad_cla
defenses['pad']['BLE'] = pad_ble
defenses['delay_group'] = dict()
defenses['delay_group']['CLA'] = delay_nextunit
defenses['delay_group']['BLE'] = delay_nextunit
defenses['delay_group+pad'] = dict()
defenses['delay_group+pad']['CLA'] = delay_nextunit_then_pad_cla
defenses['delay_group+pad']['BLE'] = delay_nextunit_then_pad_ble
defenses['add_dummies'] = dict()
defenses['add_dummies']['CLA'] = front_cla
defenses['add_dummies']['BLE'] = front_ble
defenses['normal'] = dict()
defenses['normal']['CLA'] = normal_cla
defenses['normal']['BLE'] = normal_ble

