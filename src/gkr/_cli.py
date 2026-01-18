import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(prog="zknotes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Example: zknotes sumcheck total-degree
    p_sum = sub.add_parser("sumcheck")
    p_sum.add_argument("which", choices=["total-degree"])

    args = parser.parse_args()

    if args.cmd == "sumcheck":
        mod = importlib.import_module("zknotes.sum_check")
        if args.which == "total-degree":
            mod.total_degree_example()
