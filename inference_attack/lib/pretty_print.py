
from tabulate import tabulate


def pad(s, length, max_length = -1):
    s = str(s)
    if max_length > -1 and len(s) > max_length:
        s = s[:max_length-3]+"..."
    if len(s) == length:
        return s
    p = " " * (length - len(s))
    return s+p


table_to_print = []

def new_table():
    global table_to_print
    table_to_print = []

def table_push(*args):
    global table_to_print
    table_to_print.append(args)

def table_print(headers=False):
    global table_to_print
    if headers:
        print(tabulate(table_to_print[1:], headers=table_to_print[0], tablefmt="plain"))
    else:
        print(tabulate(table_to_print, tablefmt="plain"))
    print("")
    table_to_print = []