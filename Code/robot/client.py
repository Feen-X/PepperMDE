#!/usr/bin/env python
# coding: utf-8
# PYTHON_ARGCOMPLETE_OK
#
# Copyright 2021 The Technical University of Denmark
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import argcomplete
import debugpy
from robot_client import RobotClient
import sys

def read_line() -> str:
    """
    Reads a line from the server via stdin.
    """
    return sys.stdin.readline().rstrip()


def load_level_file_from_server():
    lines = []
    while True:
        line = read_line()
        lines.append(line)
        if line.startswith("#end"):
            break

    return lines


def load_level_file_from_path(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def create_parser():
    parser = argparse.ArgumentParser(
        description="Search-client for MAvis using state-space graph search.\n"
                    "Example usage:\n"
                    "  python3 client.py classic --strategy bfs\n"
                    "  python3 client.py robot --ip 192.168.1.100 --strategy astar --heuristic goalcount",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Global options
    parser.add_argument('--max-memory', metavar="<GB>", type=str, default="4g",
                        help="The maximum memory allowed (e.g., 4g for 4 GB).")
    parser.add_argument('--debug', action='store_true',
                        default=False, help='Enable debug mode.')
    parser.add_argument("--level", type=str, default="",
                        help="Load level file from the filesystem instead of the server")
    # Action library selection
    parser.add_argument(
        "--action-library",
        choices=["default"],
        default="default",
        help="Select the action library. Default is 'default'.",
    )

    subparsers = parser.add_subparsers(
        dest="agent_type",
        required=True,
        help="Select the agent type to use",
    )

    # Strategy-related arguments
    strategy_parent = argparse.ArgumentParser(add_help=False)
    strategy_parent.add_argument(
        "--strategy",
        choices=["bfs", "dfs", "astar", "greedy"],
        default="bfs",
        help="Select the search strategy. Default is BFS."
    )
    strategy_parent.add_argument(
        "--heuristic",
        choices=["goalcount", "advanced"],
        help="Select the heuristic (only relevant for A* and Greedy)."
    )
    
    # And-Or-Graph-Search arguments
    and_or_graph_search_parent = argparse.ArgumentParser(add_help=False)
    and_or_graph_search_parent.add_argument(
        "--no-iterative-deepening",
        action="store_true",
        default=False,
        help="Disable iterative deepening"
    )
    and_or_graph_search_parent.add_argument(
        "--cyclic",
        action="store_true",
        default=False,
        help="Allow cyclic solutions"
    )
    
    # Classic agent subcommand
    classic_parser = subparsers.add_parser(
        "classic",
        help="Use a classic centralized agent using graph search",
        parents=[strategy_parent]
    )

    # Decentralised agent subcommand
    decentralised_parser = subparsers.add_parser(
        "decentralised",
        help="Use a decentralised planning agent",
        parents=[strategy_parent]
    )

    # Helper agent subcommand
    helper_parser = subparsers.add_parser(
        "helper",
        help="Use a helper agent",
        parents=[strategy_parent]
    )

    # Non-deterministic agent subcommand
    nondet_parser = subparsers.add_parser(
        "nondeterministic",
        help="Use a non-deterministic agent using AND-OR graph search",
        parents=[and_or_graph_search_parent]
    )

    # Goal recognition agent subcommand
    goalrec_parser = subparsers.add_parser(
        "goalrecognition",
        help="Use a goal recognition agent using the all optimal plans for the actor and AND-OR-GRAPH-SEARCH for the helper",
        parents=[strategy_parent, and_or_graph_search_parent]
    )

    # Robot agent subcommand
    robot_parser = subparsers.add_parser(
        "robot",
        help="A planning agent which forwards the actions to a connected pepper robot.",
        parents=[strategy_parent]
    )
    robot_parser.add_argument(
        "--ip", type=str, required=True, help="IP address of the physical robot"
    )

    return parser


def main():
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("localhost", 1234))
        debugpy.wait_for_client()
        debugpy.breakpoint()


if __name__ == "__main__":
    main()
