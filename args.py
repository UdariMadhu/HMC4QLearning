import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sNum", type=float, default=1000, help="Number of states")
parser.add_argument("--sDim", type=float, default=5, help="NDimension of state space")

parser.add_argument("--aNum", type=float, default=100, help="Number of actions")
parser.add_argument("--aDim", type=float, default=5, help="NDimension of action space")
