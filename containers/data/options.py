#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeScale", type=float, help="time of scale tensor", default=0.5)
    parser.add_argument("--timeScaleOfExtend", type=float, help="time of extend tensor", default=0.5)
    parser.add_argument("--timeScaleONarrow", type=float, help="time of narrow tensor", default=0.5)
    parser.add_argument("--timeSwap", type=float, help="time of swap tensors (0-0.001s)", default=0)
    parser.add_argument("--timeDeleteTensor", type=float, help="time of delete tensors (0-0.001s)", default=0)
    parser.add_argument("--timeAddTensor", type=float, help="time of add tensors", default=1)
    parser.add_argument("--INFINITY", type=int, help="INFINITY", default=1e1000)
    args = parser.parse_args()
    return args

