from itertools import product
from typing import Optional, Set, Dict, Callable, Tuple, Any, List
import ast
from sympy import Symbol, Add, Mul, Pow, preorder_traversal, srepr, Expr, symbols, Poly
from sympy import isprime
from sympy.polys.domains import GF
from sympy.polys.domains.finitefield import FiniteField
from graphviz import Digraph
import re
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import from_pydot
from collections import defaultdict
from utils import print_header, RED, GREEN, YELLOW, PINK, BLUE, PURPLE, RESET
import copy
import random
from multilinear_extensions import multilinear_extension


class ArithmeticCircuit:
    """
    Represents an arithmetic circuit constructed from mathematical expressions.

    This class enables the creation, analysis, and manipulation of arithmetic circuits represented as directed acyclic graphs (DAGs).
    It integrates functionality for visualization, gate evaluation, and wiring predicate computation using Graphviz and NetworkX libraries.

    Attributes:
        graphviz_circuit (Digraph): A Graphviz Digraph representation of the circuit, created from input expressions.
        networkx_circuit_machine_ID (nx.DiGraph): A NetworkX DiGraph representation of the circuit with machine-generated node IDs.
        is_dag (bool): Indicates whether the circuit is a Directed Acyclic Graph (DAG). If False, the circuit is invalid.
        fan_in_two (bool): Indicates whether all non-source nodes in the circuit have an in-degree of 2. If False, mathematical results may not apply.
        size (int): Total number of nodes in the circuit graph.
        depth (int): Depth of the circuit graph, measured as the longest path from sinks to sources.
        networkx_circuit (nx.DiGraph): A relabeled NetworkX representation of the circuit with human-readable node IDs.
        node_dict (dict): A mapping from original machine-generated node IDs to human-readable bitstring-based IDs.
        graphviz_circuit_bitstring (Digraph): A Graphviz Digraph representation with node labels replaced by bitstrings.
        layers (dict): A dictionary mapping each layer index to the set of nodes in that layer.
        topological_ordering (list): A topological ordering of nodes in the NetworkX circuit, useful for evaluating the circuit layer by layer.
        W_dict (dict): A dictionary storing the values of each gate in the circuit, layer by layer.
        add_dict (dict): A dictionary representing the addition wiring predicates for each layer.
        mult_dict (dict): A dictionary representing the multiplication wiring predicates for each layer.
        W (dict): A dictionary of functions computing the values of each gate in the circuit for given inputs.
        tilde_W (dict): A dictionary of multilinear extensions of the gate-value functions (W).
        add (dict): A dictionary of functions representing the addition wiring predicates.
        tilde_add (dict): A dictionary of multilinear extensions of the addition wiring predicates.
        mult (dict): A dictionary of functions representing the multiplication wiring predicates.
        tilde_mult (dict): A dictionary of multilinear extensions of the multiplication wiring predicates.

    Methods:
        print_topological_order(): Prints the topological ordering of the nodes in the circuit.
        print_gate_values(): Prints the values assigned to each gate (e.g., W_2(0,0) = 4).
        print_wiring_predicates(): Prints the nonzero wiring predicates for addition and multiplication.
        print_tilde_W(): Prints the multilinear extensions of the gate-value functions (W).
        print_wiring_predicate_mles(): Prints the multilinear extensions of the addition and multiplication wiring predicates.
        print_verification_propagation_equation(mle: Optional[bool]): Prints the layer-wise gate value propagation equation or its multilinear extension version.

    Usage:
        - Input a list of mathematical expressions to construct the circuit.
        - Visualize the circuit using the Graphviz representation.
        - Compute and analyze gate values and wiring predicates.
        - Verify Thaler's identity or the propagation equation.
    """

    def __init__(self, *expr_strings: Optional[str], prime: int):
        # 0. Underlying field, which must be a prime field
        if not isprime(prime):
            raise ValueError(f"We can only work with prime fields. Enter a prime number for the order of the field.\n")
        else:
            self.field = GF(prime, symmetric=False)
        # 1. Create the Graphviz circuit
        self.graphviz_circuit = create_arithmetic_circuit(*expr_strings)

        # 2. Convert Graphviz circuit to NetworkX graph
        self.networkx_circuit_machine_ID = dot_source_to_networkx(self.graphviz_circuit.source)

        # 3. Check if the graph is a DAG
        self.is_dag = nx.is_directed_acyclic_graph(self.networkx_circuit_machine_ID)
        if not self.is_dag:
            print(f"\nWARNING: The purported circuit is not a directed acyclic graph. It therefore does not model a well-defined computation.\n")

        # 4. Check if all non-source nodes have fan-in of 2
        self.fan_in_two = check_fan_in_two(self.networkx_circuit_machine_ID)
        if not self.fan_in_two:
            print(f"\nWARNING: The circuit contains gates with a fan-in not equal to 2. The mathematical results and proofs may not be valid for this configuration.\n")

        # 5. Size of the graph (number of nodes)
        self.size = self.networkx_circuit_machine_ID.number_of_nodes()

        # 6. Depth of the graph
        self.depth = find_graph_depth_dag(self.networkx_circuit_machine_ID) if self.is_dag else None

        # 7. Relabel nodes and get the mapping
        self.networkx_circuit, self.node_dict = relabel_nodes_with_layers_and_get_map(self.networkx_circuit_machine_ID)

        # 8. Create new graphviz_circuit that is a copy of the original but with bitstring node labels.
        self.graphviz_circuit_bitstring = deep_copy_and_add_labels(self.graphviz_circuit,self.node_dict)

        # 9. Partition nodes into layers
        self.layers = partition_layers(self.networkx_circuit)

        # Check structure of circuit
        check_unique_layer_assignment(self.networkx_circuit, self.layers)
        check_layer_structure(self.networkx_circuit, self.layers)

        # 10. Compute the topological ordering as an attribute
        self.topological_ordering = list(nx.topological_sort(self.networkx_circuit)) if self.is_dag else None

        # 11. Create wiring predicates. First step, create W_dict
        try:
            self.W_dict, self.add_dict, self.mult_dict = construct_W_and_wiring_dicts(self.networkx_circuit_machine_ID, self.node_dict, self.layers,self.field.mod)
        except Exception as e:
            print(f"Unable to construct gate values or wiring predicates: {e}")
        # Now create the actual functions and their multilinear extensions
        try:
            self.W, self.tilde_W = generate_layer_functions_and_extensions(self.W_dict, self.field,multilinear_extension)
            self.add, self.tilde_add, self.mult, self.tilde_mult = generate_add_mult_functions_and_extensions(self.add_dict, self.mult_dict, self.field, multilinear_extension)
        except Exception as e:
            print(f"Unable to construct multilinear extensions of gate values or wiring predicates: {e}.")
    def print_topological_order(self) -> None:
        """
        Prints the topological ordering of the nodes in the circuit.
        """
        if not self.is_dag:
            raise ValueError(f"{RED}\nThe provided graph is not a directed acyclic graph (DAG).{RESET}\n")

        print_header(f"{PINK}Topological ordering{RESET}\n", level=2)
        print(f" {YELLOW}->{RESET} ".join(self.topological_ordering))

    def print_gate_values(self) -> None:
        """
        Prints the value assigned to each gate, e.g. W_2(0,0) = 4 and so on
        """
        print_W_dict(self.W_dict)

    def print_wiring_predicates(self) -> None:
        """
        Prints nonzero wiring predicates
        """
        print_add_mult_dicts(self.add_dict, self.mult_dict)

    def print_tilde_W(self):
        """
        Prints multilinear extensions of gate-value functions W
        """
        print_multilinear_extensions(self.W_dict, self.tilde_W)

    def print_wiring_predicate_mles(self):
        print_header("Multlinear extensions of wiring predicates: ADD\n", level=2)

        tilde_add = self.tilde_add
        tilde_mult = self.tilde_mult
        W_dict = self.W_dict

        for i in reversed(sorted(tilde_add.keys())):
            print(f"Layer {i}\n".upper())
            v = len(next(iter(W_dict[i].keys())))
            w = len(next(iter(W_dict[i + 1].keys())))
            formatted_input = ','.join(
                [f"z_{j}" for j in range(v)] + [f"x_{j}" for j in range(w)] + [f"y_{j}" for j in range(w)])
            LHS = f"add\u0303_{i}({formatted_input})"
            RHS = replace_symbols_in_polynomial(tilde_add[i], v, w).as_expr()
            print(f"{LHS} = {RHS}")
            print("")

        print_header("Multlinear extensions of wiring predicates: MULT\n", level=2)

        for i in reversed(sorted(tilde_mult.keys())):
            print(f"Layer {i}\n".upper())
            v = len(next(iter(W_dict[i].keys())))
            w = len(next(iter(W_dict[i + 1].keys())))
            formatted_input = ','.join(
                [f"z_{j}" for j in range(v)] + [f"x_{j}" for j in range(w)] + [f"y_{j}" for j in range(w)])
            LHS = f"mult\u0303_{i}({formatted_input})"
            RHS = replace_symbols_in_polynomial(tilde_mult[i], v, w).as_expr()
            print(f"{LHS} = {RHS}")
            print("")

    def print_verification_propagation_equation(self, mle: Optional[bool] = None,
                                                random_selection: Optional[int] = None):
        if mle is None:
            mle = False
        if mle:
            W, W_dict, add, mult, F = self.tilde_W, self.W_dict, self.tilde_add, self.tilde_mult, self.field
            print_header(f"Verification of Thaler's identity\n", level=2)
        else:
            W, W_dict, add, mult, F = self.W, self.W_dict, self.add, self.mult, self.field
            print_header(f"Verification of layer-wise gate-value propagation equation\n", level=2)

        for i in sorted(add.keys(), reverse=True):  # Iterate over layers in descending order
            print(f"Layer {i}\n".upper())
            v = len(next(iter(W_dict[i].keys())))
            w = len(next(iter(W_dict[i + 1].keys())))

            if mle:
                z_range = list(product(range(self.field.mod), repeat=v))
            else:
                z_range = list(product([0, 1], repeat=v))

            if random_selection:
                if random_selection < len(z_range):
                    print(f"(Random selection of {random_selection} z-values.)\n")
                z_range = random.sample(z_range, min(random_selection, len(z_range)))

            for z in z_range:
                if mle:
                    LHS = F(W[i].eval(z))
                else:
                    LHS = F(W[i](z))
                LHS = int(LHS)

                xy_range = list(product(product([0, 1], repeat=w), repeat=2))

                if mle:
                    RHS = sum(
                        [add[i].eval(z + x + y) * (W[i + 1].eval(x) + W[i + 1].eval(y)) +
                         mult[i].eval(z + x + y) * W[i + 1].eval(x) * W[i + 1].eval(y)
                         for x, y in xy_range]
                    )
                else:
                    RHS = sum(
                        [add[i](z + x + y) * (W[i + 1](x) + W[i + 1](y)) +
                         mult[i](z + x + y) * W[i + 1](x) * W[i + 1](y)
                         for x, y in xy_range]
                    )
                RHS = F(RHS)
                RHS = int(RHS)

                if LHS == RHS:
                    correct = f"{GREEN}\u2713{RESET}"  # tick
                else:
                    correct = f"{RED}\u2717{RESET}"  # cross

                formatted_tuple = ','.join([f"{z_j}" for z_j in z])
                if mle:
                    formatted_LHS = f"W\u0303_{i}({formatted_tuple})"
                    formatted_RHS = (
                        f"sum {{ add\u0303_{i}(({formatted_tuple}),x,y) [ W\u0303_{i + 1}(x) + W\u0303_{i + 1}(y) ] + "
                        f"mult\u0303_{i}({formatted_tuple},x,y) [ W\u0303_{i + 1}(x) W\u0303_{i + 1}(y)] }} over (x,y) in "
                        f"{{0,1}}^{v} × {{0,1}}^{v}")
                else:
                    formatted_LHS = f"W_{i}({formatted_tuple})"
                    formatted_RHS = (f"sum {{ add_{i}(({formatted_tuple}),x,y) [ W_{i + 1}(x) + W_{i + 1}(y) ] + "
                                     f"mult_{i}({formatted_tuple},x,y) [ W_{i + 1}(x) W_{i + 1}(y)] }} over (x,y) in "
                                     f"{{0,1}}^{v} × {{0,1}}^{v}")

                print(f"{formatted_LHS} = {LHS}, {formatted_RHS} = {RHS} {correct}")
            print("")

"""
START: CONSTRUCT ARITHMETIC CIRCUIT FROM STRING EXPRESSION
"""
def parse_expression(expr_str: str) -> Expr:
    """
    Parse a string expression into a SymPy expression with integers replaced by symbols.

    Args:
        expr_str (str): The arithmetic expression as a string (e.g., '(2 + 2)*(3*3)').

    Returns:
        Expr: The SymPy expression with integers replaced by symbols.
    """
    # Parse the string into an Abstract Syntax Tree (AST)
    expr_ast = ast.parse(expr_str, mode='eval')
    # Recursively convert the AST into a SymPy expression
    expr_sympy = ast_to_sympy(expr_ast.body)
    return expr_sympy

def ast_to_sympy(node: ast.AST) -> Expr:
    """
    Recursively convert an AST node into a SymPy expression, replacing integers with symbols.

    Args:
        node (ast.AST): The AST node to convert.

    Returns:
        Expr: The corresponding SymPy expression.

    Raises:
        ValueError: If an unsupported AST node or operator is encountered.
    """
    if isinstance(node, ast.Constant):  # For Python 3.8 and above
        # Handle numeric constants
        if isinstance(node.value, int):
            # Replace integer constants with symbols named after the integers
            return Symbol(str(node.value), commutative=False)
        elif isinstance(node.value, float):
            # Replace float constants with symbols named after the floats
            return Symbol(str(node.value), commutative=False)
        else:
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.BinOp):
        # Handle binary operations (e.g., addition, subtraction, multiplication)
        left = ast_to_sympy(node.left)   # Convert the left operand
        right = ast_to_sympy(node.right) # Convert the right operand
        op = node.op                     # Get the operator

        if isinstance(op, ast.Add):
            # Addition operation
            return Add(left, right, evaluate=False)
        elif isinstance(op, ast.Sub):
            # Subtraction operation: a - b => a + (-1)*b
            return Add(left, Mul(-1, right, evaluate=False), evaluate=False)
        elif isinstance(op, ast.Mult):
            # Multiplication operation
            return Mul(left, right, evaluate=False)
        elif isinstance(op, ast.Div):
            # Division operation: a / b => a * b^(-1)
            return Mul(left, Pow(right, -1, evaluate=False), evaluate=False)
        elif isinstance(op, ast.Pow):
            # Exponentiation operation
            return Pow(left, right, evaluate=False)
        else:
            # Unsupported binary operator
            raise ValueError(f"Unsupported binary operator: {op}")

    elif isinstance(node, ast.UnaryOp):
        # Handle unary operations (e.g., negation)
        operand = ast_to_sympy(node.operand)  # Convert the operand
        op = node.op                          # Get the operator

        if isinstance(op, ast.UAdd):
            # Unary plus (e.g., +a)
            return operand
        elif isinstance(op, ast.USub):
            # Unary minus (e.g., -a)
            return Mul(-1, operand, evaluate=False)
        else:
            # Unsupported unary operator
            raise ValueError(f"Unsupported unary operator: {op}")

    elif isinstance(node, ast.Name):
        # Handle variable names
        return Symbol(node.id, commutative=False)

    elif isinstance(node, ast.Expr):
        # Handle expression nodes
        return ast_to_sympy(node.value)

    else:
        # Unsupported AST node type
        raise ValueError(f"Unsupported AST node type: {type(node)}")

def build_circuit(expr: Expr, graph: Digraph, parent: Optional[str] = None) -> str:
    """
    Recursively builds a graph representation of a SymPy expression.

    Args:
        expr (Expr): The SymPy expression to be converted into a graph.
        graph (Digraph): The graph object to which nodes and edges will be added.
        parent (str, optional): The parent node ID in the graph (used for recursive calls).

    Returns:
        str: The node ID of the current expression in the graph.
    """
    if expr.is_Atom:  # Base case: variable or constant
        node_id = str(id(expr))
        label = str(expr)
        graph.node(node_id, label=label, shape="circle")
        if parent:
            graph.edge(node_id, parent)
        return node_id

    # For compound expressions
    op_name = type(expr).__name__

    # Map operation names to desired labels
    if op_name == 'Add':
        label = '+'
    elif op_name == 'Mul':
        label = '×'  # Using the multiplication symbol
    else:
        label = op_name

    node_id = str(id(expr))
    graph.node(node_id, label=label, shape="circle")  # Use circle shape for operators

    if parent:
        graph.edge(node_id, parent)

    # Recurse on arguments of the expression
    for arg in expr.args:
        build_circuit(arg, graph, node_id)
    return node_id

def create_arithmetic_circuit(*expr_strings: str) -> Digraph:
    """
    Creates an arithmetic circuit graph from multiple string expressions.

    Args:
        *expr_strings (str): Arbitrary number of string expressions to be included in the circuit.

    Returns:
        Digraph: The graph object representing the arithmetic circuits.
    """
    # Initialize the graph
    graph = Digraph(comment="Arithmetic Circuit", format="png")
    graph.attr(rankdir="BT")  # Set the graph direction from bottom to top

    # Loop over each expression string
    for expr_str in expr_strings:
        # Parse the expression
        expr = parse_expression(expr_str)
        # Build the circuit graph for this expression
        build_circuit(expr, graph)
        # Optionally, add a separator node or visual separation if needed

    return graph

# Example usage
if __name__ == "__main__":
    # Create the arithmetic circuit with multiple expressions
    dot = create_arithmetic_circuit(
        '(2 + 2)*(3*3)',
        '(3*4) + (5 + 5)'
    )
    # Render the graph to a file and display it
    dot.render('arithmetic_circuit', view=True)

"""
END: CONSTRUCT ARITHMETIC CIRCUIT FROM STRING EXPRESSION
"""

"""
START: PARTITIONING ARITHMETIC CIRCUIT INTO LAYERS, LABELING GATES WITH BITSTRINGS, ETC.
"""

def parse_dot_source(dot_source: str):
    """
    Parse a DOT source string to extract nodes, edges, and their attributes.

    Args:
        dot_source (str): The DOT source code as a string.

    Returns:
        dict: A dictionary with 'nodes' and 'edges'.
            - 'nodes': A dictionary mapping node IDs to their attributes (label, shape).
            - 'edges': A list of tuples representing edges (tail_node, head_node).
    """
    # Patterns for extracting nodes and edges
    node_pattern = re.compile(r'^\s*(\w+)\s*\[label="([^"]+)"\s+shape=([^]]+)\];?$', re.MULTILINE)
    edge_pattern = re.compile(r'^\s*(\w+)\s*->\s*(\w+)\s*;?$', re.MULTILINE)

    # Extract nodes with attributes
    nodes = {}
    for match in node_pattern.findall(dot_source):
        node_id, label, shape = match
        nodes[node_id] = {'label': label, 'shape': shape}

    # Extract edges
    edges = edge_pattern.findall(dot_source)

    return {'nodes': nodes, 'edges': edges}

def dot_source_to_networkx(dot_source: str) -> nx.DiGraph:
    """
    Converts a DOT source string to a NetworkX directed graph.

    Args:
        dot_source (str): The DOT source code as a string.

    Returns:
        nx.DiGraph: The corresponding NetworkX directed graph.
    """
    # Parse the DOT source into a Pydot graph
    pydot_graph = pydot.graph_from_dot_data(dot_source)[0]

    # Convert the Pydot graph to a NetworkX graph
    nx_graph = from_pydot(pydot_graph)
    return nx_graph

def check_fan_in_two(nx_graph: nx.DiGraph) -> bool:
    """
    Checks the in-degree of each node in the graph. If any non-input node
    has an in-degree other than 2, returns False. Otherwise, returns True

    Args:
        nx_graph (nx.DiGraph): A directed graph representing the arithmetic circuit.

    Returns:
        True or False
    """
    for node in nx_graph.nodes:
        indegree = nx_graph.in_degree(node)
        if indegree == 0:
            # Input nodes are expected to have in-degree 0
            continue
        elif indegree != 2:
            # Non-input nodes must have in-degree 2
            return False
    return True

def find_graph_depth_dag(nx_graph: nx.DiGraph) -> int:
    """
    Finds the depth of a directed acyclic graph (DAG).

    Args:
        nx_graph (nx.DiGraph): A directed acyclic graph.

    Returns:
        int: The depth (longest path length) of the graph.
    """
    # Ensure the graph is a DAG
    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ValueError("The graph is not a directed acyclic graph (DAG).")

    # Perform a topological sort
    topo_order = list(nx.topological_sort(nx_graph))

    # Compute the longest path
    longest_path_length = {node: 0 for node in topo_order}
    for node in topo_order:
        for successor in nx_graph.successors(node):
            longest_path_length[successor] = max(
                longest_path_length[successor],
                longest_path_length[node] + 1
            )

    return max(longest_path_length.values())

def print_topological_ordering(nx_graph: nx.DiGraph):
    """
    Prints the topological ordering of nodes in a directed acyclic graph (DAG).

    Args:
        nx_graph (nx.DiGraph): A directed acyclic graph.

    Raises:
        ValueError: If the graph is not a DAG.
    """
    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ValueError(f"{RED}\nThe provided graph is not a directed acyclic graph (DAG).{RESET}\n")

    # Get the topological ordering
    topo_order = list(nx.topological_sort(nx_graph))

    # Print the ordering
    print_header(f"{PINK}Topological ordering{RESET}\n", level=2)
    print(f" {YELLOW}->{RESET} ".join(topo_order))


def partition_layers(nx_graph: nx.DiGraph) -> dict[int, set[str]]:
    """
    Partitions the nodes of a graph into layers based on their distance from all sink nodes.
    The layers are indexed by distance, and each layer contains the union of nodes at that distance
    from any sink.

    Args:
        nx_graph (nx.DiGraph): A directed graph.

    Returns:
        dict[int, set[str]]: A dictionary where the keys are distances (layer indices)
                             and the values are sets of nodes in each layer.
    """
    # Find all sink nodes
    sinks = [node for node in nx_graph.nodes if nx_graph.out_degree(node) == 0]

    # Initialize a dictionary to store the union of layers across all sinks
    combined_layers = defaultdict(set)

    # Process each sink node
    for sink in sinks:
        # Reverse the graph to measure distances toward the sink
        reversed_graph = nx.reverse(nx_graph)

        # Compute shortest path lengths from the sink
        distances = nx.single_source_shortest_path_length(reversed_graph, source=sink)

        # Add nodes to the appropriate layer
        for node, distance in distances.items():
            combined_layers[distance].add(node)

    # Convert defaultdict to a regular dict and return
    return dict(combined_layers)


def check_layer_structure(nx_graph: nx.DiGraph, layers: dict[int, set[str]]) -> bool:
    """
    Checks if the graph satisfies the following properties:

    For every node in layer i:
    - Layer 0 (output layer) has incoming edges only from layer 1.
    - Layer i (for 1 <= i < d-1) has incoming edges only from layer i+1.
    - Layer d-1 (input layer) has no incoming edges.

    Args:
        nx_graph (nx.DiGraph): A directed graph.
        layers (dict[int, set[str]]): A dictionary of layers, where keys are the layer indices and the values are sets
                                      of nodes in each layer.

    Returns:
        bool: True if the graph satisfies the condition, False otherwise.
    """
    # Find the maximum layer index (the input layer)
    max_layer = max(layers.keys())

    # (1) Check that the output layer (layer 0) has incoming edges only from layer 1
    for u in layers[0]:  # The output layer is layer 0
        for pred in nx_graph.predecessors(u):
            if pred not in layers[1]:
                print(f"Warning! Circuit does not have desired structure: node {u} in layer 0 has an incoming edge from node {pred} which is not in layer 1.")
                return False

    # (2) Check that layers i (for 1 <= i < max_layer-1) have incoming edges only from layer i+1
    for i in range(1, max_layer):  # Skip the first and last layers
        for u in layers[i]:  # For each node u in layer i
            for pred in nx_graph.predecessors(u):
                if pred not in layers[i + 1]:
                    print(
                        f"Warning! Circuit does not have desired structure: node {u} in layer {i} has an incoming edge from node {pred} which is not in layer {i + 1}.")
                    return False

    # (3) Check that the input layer (max_layer) has no incoming edges
    for u in layers[max_layer]:  # The input layer is layer max_layer
        if nx_graph.in_degree(u) > 0:
            print(f"Warning! Circuit does not have desired structure: node {u} in input layer ({max_layer}) has incoming edges.")
            return False

    # If all checks pass, return True
    return True


def check_unique_layer_assignment(nx_graph: nx.DiGraph, layers: dict[int, set[str]]) -> bool:
    """
    Checks if every node appears in exactly one layer in the given layer partition.

    Args:
        nx_graph (nx.DiGraph): A directed graph.
        layers (dict[int, set[str]]): A dictionary of layers, where keys are the layer indices and the values are sets
                                      of nodes in each layer.

    Returns:
        bool: True if each node appears in exactly one layer, False otherwise.
    """
    # Set to track nodes we have seen across layers
    seen_nodes = set()

    # Iterate over each layer and its nodes
    for layer, nodes in layers.items():
        for node in nodes:
            # If the node has already been seen, it appears in more than one layer
            if node in seen_nodes:
                print(f"Warning! Circuit does not have desired structure: node {node} appears in multiple layers.")
                return False
            seen_nodes.add(node)

    # Ensure that all nodes in the graph are accounted for in the layers
    all_nodes_in_layers = set(node for nodes in layers.values() for node in nodes)
    graph_nodes = set(nx_graph.nodes)

    if all_nodes_in_layers != graph_nodes:
        missing_nodes = graph_nodes - all_nodes_in_layers
        extra_nodes = all_nodes_in_layers - graph_nodes
        if missing_nodes:
            print(f"Warning! Circuit does not have desired structure: the following nodes are missing from the layers: {missing_nodes}")
        if extra_nodes:
            print(f"Warning! Circuit does not have desired structure: the following nodes are extra in the layers: {extra_nodes}")
        return False

    # If all checks pass, return True
    return True

def relabel_nodes_with_layers_and_get_map(nx_graph: nx.DiGraph) -> tuple[nx.DiGraph, dict]:
    """
    Relabel the nodes of a graph using the scheme:
    'i.bbbb...b', where 'i' is the layer index and 'bbbb...b' is the
    zero-padded binary representation of the node's position within the layer.

    In the degenerate case where the layer contains only one node, the bitstring is '_'.

    Args:
        nx_graph (nx.DiGraph): A directed graph.

    Returns:
        tuple[nx.DiGraph, dict]: A new graph with nodes relabeled and the mapping of old IDs to new IDs.
    """
    layers = partition_layers(nx_graph)  # Get layers
    relabel_map = {}  # Initialize relabeling map

    for i, nodes in layers.items():
        sorted_nodes = sorted(nodes)  # Sort the nodes
        if len(sorted_nodes) == 1:  # Degenerate case: single node in the layer
            bitstring = "_"
            node = next(iter(sorted_nodes))  # Only one node exists
            new_id = f"{i}.{bitstring}"  # Create new ID
            relabel_map[node] = new_id  # Map old to new ID
        else:  # General case: multiple nodes in the layer
            bitstring_length = len(bin(len(sorted_nodes) - 1)[2:])  # Max bitstring length
            for idx, node in enumerate(sorted_nodes):
                bitstring = bin(idx)[2:].zfill(bitstring_length)  # Convert to binary and pad
                new_id = f"{i}.{bitstring}"  # Create new ID
                relabel_map[node] = new_id  # Map old to new ID

    # Relabel the graph nodes
    relabeled_graph = nx.relabel_nodes(nx_graph, relabel_map)
    return relabeled_graph, relabel_map

def relabel_dot_source(dot_source: str, relabel_map: dict) -> str:
    """
    Relabels the nodes in a DOT source string according to the provided mapping.

    Args:
        dot_source (str): The original DOT source string.
        relabel_map (dict): A dictionary mapping old node IDs to new node IDs.

    Returns:
        str: The relabeled DOT source string.
    """
    # Replace old node IDs with new IDs in the DOT source
    for old_id, new_id in relabel_map.items():
        dot_source = dot_source.replace(str(old_id), str(new_id))
    return dot_source

def deep_copy_and_add_labels(dot_obj: Digraph, new_labels: dict, font_size: int = 10) -> Digraph:
    """
    Deep copies a Graphviz Digraph object and adds new labels to its nodes.

    Args:
        dot_obj (Digraph): The original Graphviz Digraph.
        new_labels (dict): A dictionary mapping old node IDs to new labels.
        font_size (int): Font size for the node labels.

    Returns:
        Digraph: A new Graphviz Digraph with updated node labels.
    """
    # Step 1: Create a deep copy of the original graph
    copied_dot = Digraph(name=dot_obj.name, comment=dot_obj.comment, format=dot_obj.format)

    # Set the graph orientation to Bottom-to-Top (BT)
    copied_dot.attr(rankdir="BT")

    # Step 2: Extract old labels from the original dot source
    old_labels = {}
    label_pattern = r'label="([^"]*)"|label=([^ ]*)'
    matches = re.findall(label_pattern, dot_obj.source)
    for line in dot_obj.source.splitlines():
        if '[' in line and 'label=' in line:  # Node definition line with a label
            node_id = line.split()[0]
            label_match = re.search(label_pattern, line)
            if label_match:
                # Extract the label from the match
                old_label = label_match.group(1) if label_match.group(1) else label_match.group(2)
                old_labels[node_id] = old_label

    # Step 3: Add updated nodes with new labels
    for node_id, new_label in new_labels.items():
        if node_id in old_labels:
            old_label = old_labels[node_id]
            new_label_part = str(new_label).split('.')[-1]  # Ensure new_label is a string and take the part after the `.`

            # Combine old_label and new_label
            # combined_label = f"{old_label}\n{new_label_part}"
            combined_label = f"""<<FONT COLOR="black">{old_label}</FONT><BR/><FONT COLOR="blue">{new_label_part}</FONT>>"""

            # Determine shape based on new_label length
            shape = "ellipse" if len(new_label_part) > 3 else "circle"

            # Add the updated node to the copied graph
            copied_dot.node(node_id, label=combined_label, shape=shape, fontsize=str(font_size))

    # Step 4: Preserve edges
    for line in dot_obj.source.splitlines():
        if '->' in line:  # Edge definition
            copied_dot.body.append(line)  # Preserve edges

    return copied_dot

"""
START: PARTITIONING ARITHMETIC CIRCUIT INTO LAYERS, LABELING GATES WITH BITSTRINGS, ETC.
"""

"""
START: WIRING PREDICATES
"""

def construct_W_and_wiring_dicts(
    machine_id_graph: nx.DiGraph,
    id_mapping: Dict[str, str],
    layer_partition: Dict[int, Set[str]],
    p: Optional[int] = None,
) -> Tuple[
    Dict[int, Dict[Tuple[Any, ...], int]],
    Dict[int, Dict[Tuple[Any, ...], int]],
    Dict[int, Dict[Tuple[Any, ...], int]],
]:
    """
    Constructs the nested dictionary W_dict along with add_dict and mult_dict.

    Handles the empty bitstring case by including it explicitly in the keys.
    """
    W_dict = {}
    add_dict = defaultdict(dict)
    mult_dict = defaultdict(dict)

    # Reverse the id_mapping to map bitstring IDs to machine IDs
    reverse_mapping = {v: k for k, v in id_mapping.items()}

    # Process layers in reverse order (from outputs to inputs)
    for layer in sorted(layer_partition.keys(), reverse=True):
        W_dict[layer] = {}  # Initialize dictionary for this layer

        for node in layer_partition[layer]:
            # Handle the bitstring
            raw_bitstring = node.split('.')[1]
            bitstring = tuple() if raw_bitstring == '_' else tuple(map(int, raw_bitstring))

            # Get the corresponding machine ID
            machine_id = reverse_mapping[node]

            # Normalize the label by stripping quotes
            operation = machine_id_graph.nodes[machine_id]['label'].strip('"')

            # Special case for input layer
            if layer == max(layer_partition.keys()):  # Input layer
                value = int(operation)  # Use the label directly as an integer
                W_dict[layer][bitstring] = value % p if p else value

                # Ensure all bitstrings of the given length are included
                bitstring_length = len(next(iter(W_dict[layer]))) if W_dict[layer] else 0

                # Generate all possible bitstrings of the determined length
                all_bitstrings = [tuple(bits) for bits in product(range(2), repeat=bitstring_length)]

                # Add missing bitstrings with a default value of 0
                for bitstring in all_bitstrings:
                    if bitstring not in W_dict[layer]:
                        W_dict[layer][bitstring] = 0

            else:  # Non-input layer
                # Get incoming edges
                incoming_edges = list(machine_id_graph.in_edges(machine_id))
                if len(incoming_edges) < 2:
                    raise ValueError(f"Node {node} in layer {layer} does not have at least 2 incoming edges.")

                # Map parents to bitstring-based IDs and retrieve their values
                parent_bitstrings = []
                parent_values = []
                for parent, _ in incoming_edges:
                    raw_parent_bitstring = id_mapping[parent].split('.')[1]
                    parent_bitstring = tuple() if raw_parent_bitstring == '_' else tuple(map(int, raw_parent_bitstring))
                    parent_bitstrings.append(parent_bitstring)

                    if parent_bitstring not in W_dict[layer + 1]:
                        raise KeyError(
                            f"Parent node {parent} with bitstring {id_mapping[parent]} (layer {layer + 1}) "
                            f"is missing in W_dict[layer + 1]: {list(W_dict[layer + 1].keys())}"
                        )
                    parent_values.append(W_dict[layer + 1][parent_bitstring])

                # Compute the value for the current node
                if operation == '+':
                    value = sum(parent_values)
                    key = (bitstring,) + parent_bitstrings[0] + parent_bitstrings[1]
                    add_dict[layer][key] = 1
                elif operation == '×':
                    product_value = 1
                    for v in parent_values:
                        product_value *= v
                    value = product_value
                    key = (bitstring,) + parent_bitstrings[0] + parent_bitstrings[1]
                    mult_dict[layer][key] = 1
                else:
                    raise ValueError(f"Unsupported operation {operation} for node {node}.")

                # Apply modulo if p is provided
                W_dict[layer][bitstring] = value % p if p else value

    # Ensure all layers are included when filling missing keys
    all_layers = list(layer_partition.keys())
    for layer in all_layers:
        if layer == max(layer_partition.keys()):  # Skip input layer
            continue
        # Ensure the layer exists in add_dict and mult_dict
        if layer not in add_dict:
            add_dict[layer] = {}
        if layer not in mult_dict:
            mult_dict[layer] = {}

        # Determine bitstring lengths
        bitstring_length = len(next(iter(W_dict[layer]))) if W_dict[layer] else 0
        parent_bitstring_length = len(next(iter(W_dict[layer + 1]))) if layer + 1 in W_dict else 0

        # Generate all possible combinations for keys
        if bitstring_length == 0:
            # Include the empty bitstring explicitly in the key
            all_parent_bitstrings = [tuple(bits) for bits in product(range(2), repeat=parent_bitstring_length)]
            all_possible_keys = {
                ((),) + parent_bits1 + parent_bits2
                for parent_bits1 in all_parent_bitstrings
                for parent_bits2 in all_parent_bitstrings
            }
        else:
            all_bitstrings = [tuple(bits) for bits in product(range(2), repeat=bitstring_length)]
            all_parent_bitstrings = [tuple(bits) for bits in product(range(2), repeat=parent_bitstring_length)]
            all_possible_keys = {
                (bitstring,) + parent_bits1 + parent_bits2
                for bitstring in all_bitstrings
                for parent_bits1 in all_parent_bitstrings
                for parent_bits2 in all_parent_bitstrings
            }

        for key in all_possible_keys:
            if key not in add_dict[layer]:
                add_dict[layer][key] = 0
            if key not in mult_dict[layer]:
                mult_dict[layer][key] = 0

    return W_dict, add_dict, mult_dict


def print_W_dict(W_dict: Dict[int, Dict[tuple, int]]) -> None:
    """
    Prints the W_dict in a structured format, layer by layer.

    Args:
        W_dict (Dict[int, Dict[tuple, int]]): The dictionary representing the layer-wise weights.
    """
    print_header(f"Gate values\n", level=2)
    # Iterate through layers in sorted order
    for layer in sorted(W_dict.keys(), reverse=True):  # Start from the highest layer
        print(f"Layer {layer}\n".upper())

        # Sort the tuples (keys of the inner dictionary) in lexicographical order
        sorted_keys = sorted(W_dict[layer].keys())

        # Print each weight in the layer
        for key in sorted_keys:
            formatted_key = ','.join(map(str, key))  # Convert tuple to comma-separated string
            print(f"W_{layer}({formatted_key}) = {W_dict[layer][key]}")

        print("")  # Add spacing between layers

# Helper function for print_add_mult_dicts below
def flatten_inner_tuples(lst):
    return [tuple(item if not isinstance(item, tuple) else item for sub in t for item in (sub if isinstance(sub, tuple) else [sub])) for t in lst]

def print_add_mult_dicts(add_dict: Dict[int, Dict[tuple, int]], mult_dict: Dict[int, Dict[tuple, int]]) -> None:
    """
    Prints the add_dict and mult_dict in a structured format, layer by layer.
    Only entries where the value is 1 are printed.

    Args:
        add_dict (Dict[int, Dict[tuple, int]]): The dictionary representing the addition wiring predicates.
        mult_dict (Dict[int, Dict[tuple, int]]): The dictionary representing the multiplication wiring predicates.
    """
    def filter_and_sort_dict(input_dict: Dict[int, Dict[tuple, int]]) -> Dict[int, List[tuple]]:
        """
        Filters and sorts keys with value 1 in the input_dict.

        Args:
            input_dict (Dict[int, Dict[tuple, int]]): Dictionary to process.

        Returns:
            Dict[int, List[tuple]]: Processed dictionary with sorted keys for each layer.
        """
        result = {}
        for layer in sorted(input_dict.keys(), reverse=True):
            filtered_keys = [key for key in input_dict[layer] if input_dict[layer][key] == 1]
            flattened_keys = flatten_inner_tuples(filtered_keys)
            result[layer] = sorted(flattened_keys)
        return result

    # Filter and sort both dictionaries
    add_dict_sorted = filter_and_sort_dict(add_dict)
    mult_dict_sorted = filter_and_sort_dict(mult_dict)

    # Print add_dict
    print_header("Addition Wiring Predicates (nonzero values)\n", level=2)
    for layer, keys in add_dict_sorted.items():
        print(f"Layer {layer}\n".upper())
        for key in keys:
            formatted_key = ','.join(map(str, key))
            print(f"add_{layer}({formatted_key}) = 1")
        print("")  # Add spacing between layers

    # Print mult_dict
    print_header("Multiplication Wiring Predicates (nonzero values)\n", level=2)
    for layer, keys in mult_dict_sorted.items():
        print(f"Layer {layer}\n".upper())
        for key in keys:
            formatted_key = ','.join(map(str, key))
            print(f"mult_{layer}({formatted_key}) = 1")
        print("")  # Add spacing between layers

def generate_layer_functions_and_extensions(
    W_dict: Dict[int, Dict[Tuple[int, ...], int]],
    field: FiniteField,
    multilinear_extension: Callable[[FiniteField, int, Callable[[Tuple[int, ...]], int]], Poly]
) -> Tuple[Dict[int, Callable[[Tuple[int, ...]], int]], Dict[int, Poly]]:
    """
    Generate layer functions W_i and their multilinear extensions tilde_W_i.

    Args:
        W_dict (Dict[int, Dict[Tuple[int, ...], int]]): The dictionary representing the layer-wise weights.
        field (FiniteField): The finite field over which the extensions are defined.
        multilinear_extension (Callable): A function to compute the multilinear extension.

    Returns:
        Tuple[Dict[int, Callable[[Tuple[int, ...]], int]], Dict[int, Poly]]: Dictionaries for W and tilde_W.
    """
    # Create the layer functions W_i
    def create_W(i: int) -> Callable[[Tuple[int, ...]], int]:
        def Wi(b: Tuple[int, ...]) -> int:
            return W_dict[i][b]
        return Wi

    # Initialize dictionaries for W and tilde_W
    W = {i: create_W(i) for i in W_dict.keys()}
    tilde_W = {}

    # Create the multilinear extensions for each layer
    for i in W.keys():
        # Handle the degenerate case where the keys are empty tuples
        if not W_dict[i]:  # If W_dict[i] is empty
            raise ValueError(f"W_dict for layer {i} is empty, cannot compute extensions.")

        first_key = next(iter(W_dict[i].keys()))
        v = len(first_key)  # Length of any key in layer i
        if v == 0:
            # Degenerate case: no variables in the layer
            constant_value = list(W_dict[i].values())[0]
            # Introduce a dummy variable x_0
            X = symbols(f"x_:{v + 1}")
            # Create a constant polynomial in the dummy variable
            tilde_W[i] = Poly(constant_value, X, domain=field)
        else:
            # Compute the multilinear extension
            tilde_W[i] = multilinear_extension(field, v, W[i])

    return W, tilde_W

def generate_add_mult_functions_and_extensions(
    add_dict: Dict[int, Dict[Tuple[Any, ...], int]],
    mult_dict: Dict[int, Dict[Tuple[Any, ...], int]],
    field: FiniteField,
    multilinear_extension: Callable[[FiniteField, int, Callable[[Tuple[int, ...]], int]], Poly]
) -> Tuple[
    Dict[int, Callable[[Tuple[int, ...]], int]],
    Dict[int, Poly],
    Dict[int, Callable[[Tuple[int, ...]], int]],
    Dict[int, Poly]
]:
    """
    Generate layer functions add_i, mult_i, and their multilinear extensions tilde_add_i, tilde_mult_i.

    Args:
        add_dict (Dict[int, Dict[Tuple[Any, ...], int]]): The dictionary representing addition predicates.
        mult_dict (Dict[int, Dict[Tuple[Any, ...], int]]): The dictionary representing multiplication predicates.
        field (FiniteField): The finite field over which the extensions are defined.
        multilinear_extension (Callable): A function to compute the multilinear extension.

    Returns:
        Tuple[
            Dict[int, Callable[[Tuple[int, ...]], int]],
            Dict[int, Poly],
            Dict[int, Callable[[Tuple[int, ...]], int]],
            Dict[int, Poly]
        ]:
            Dictionaries for add, tilde_add, mult, and tilde_mult.
    """
    # Create the layer functions for addition
    def create_add(i: int) -> Callable[[Tuple[int, ...]], int]:
        sample_key = next(iter(add_dict[i].keys()))
        def add_func(b: Tuple[int, ...]) -> int:
            # Reconstruct the key based on the structure of keys in add_dict[i]
            key_elements = []
            idx = 0  # Index in b
            for element in sample_key:
                if isinstance(element, tuple):
                    # For tuples (which could be empty), extract len(element) bits
                    length = len(element)
                    key_elements.append(tuple(b[idx:idx+length]))
                    idx += length
                else:
                    # For single bits
                    key_elements.append(b[idx])
                    idx += 1
            key = tuple(key_elements)
            return add_dict[i][key]
        return add_func

    # Create the layer functions for multiplication
    def create_mult(i: int) -> Callable[[Tuple[int, ...]], int]:
        sample_key = next(iter(mult_dict[i].keys()))
        def mult_func(b: Tuple[int, ...]) -> int:
            # Reconstruct the key based on the structure of keys in mult_dict[i]
            key_elements = []
            idx = 0
            for element in sample_key:
                if isinstance(element, tuple):
                    length = len(element)
                    key_elements.append(tuple(b[idx:idx+length]))
                    idx += length
                else:
                    key_elements.append(b[idx])
                    idx += 1
            key = tuple(key_elements)
            return mult_dict[i][key]
        return mult_func

    # Initialize dictionaries for add, tilde_add, mult, tilde_mult
    add = {i: create_add(i) for i in add_dict.keys()}
    tilde_add = {}
    mult = {i: create_mult(i) for i in mult_dict.keys()}
    tilde_mult = {}

    # Create the multilinear extensions for each layer
    for i in add.keys():
        if not add_dict[i]:
            continue
        sample_key = next(iter(add_dict[i].keys()))
        # Compute total number of bits v
        v = sum(len(element) if isinstance(element, tuple) else 1 for element in sample_key)
        tilde_add[i] = multilinear_extension(field, v, add[i])

    for i in mult.keys():
        if not mult_dict[i]:
            continue
        sample_key = next(iter(mult_dict[i].keys()))
        v = sum(len(element) if isinstance(element, tuple) else 1 for element in sample_key)
        tilde_mult[i] = multilinear_extension(field, v, mult[i])

    return add, tilde_add, mult, tilde_mult

def print_multilinear_extensions(
    W_dict: Dict[int, Dict[Tuple[int], int]],
    tilde_W: Dict[int, Callable]
) -> None:
    """
    Prints the multilinear extensions for each layer in a structured format.

    Args:
        W_dict (Dict[int, Dict[Tuple[int], int]]): The dictionary of layer-wise weights.
        tilde_W (Dict[int, Callable]): The dictionary of multilinear extensions for each layer.
    """
    print_header(f"Multlinear extensions of gate-value functions\n", level=2)
    for i in sorted(W_dict.keys(), reverse=True):  # Iterate over layers in descending order
        print(f"Layer {i}\n".upper())
        # Determine the length of the bitstring keys for the current layer
        v = len(next(iter(W_dict[i].keys())))
        # Generate symbolic variable names x_0, x_1, ..., x_{v-1}
        formatted_tuple = ', '.join([f"x_{j}" for j in range(v)])
        # Print the formatted output
        print(f"W\u0303_{i}({formatted_tuple}) = {tilde_W[i].as_expr()}\n")


def replace_symbols_in_polynomial(poly: Poly, v: int, w: int) -> Poly:
    """
    Replace the symbols in a SymPy polynomial.

    Args:
        poly (Poly): The input polynomial.
        v (int): Number of symbols to replace with z_j (starting at j = 0).
        w (int): Number of symbols to replace with x_j and y_j (starting at j = 0 for each).

    Returns:
        Poly: A new polynomial with replaced symbols.
    """
    # Extract symbols from the polynomial
    original_symbols = list(poly.gens)

    # Generate new symbols
    z_symbols = [symbols(f"z_{j}") for j in range(v)]
    x_symbols = [symbols(f"x_{j}") for j in range(w)]
    y_symbols = [symbols(f"y_{j}") for j in range(w)]

    # Create a mapping of original symbols to new symbols
    replacement_map = {}

    for i, sym in enumerate(original_symbols):
        if i < v:
            replacement_map[sym] = z_symbols[i]
        elif v <= i < v + w:
            replacement_map[sym] = x_symbols[i - v]
        elif v + w <= i < v + 2 * w:
            replacement_map[sym] = y_symbols[i - v - w]
        else:
            raise ValueError("Not enough replacements for the given symbols.")

    # Create a new polynomial with the replaced symbols
    replaced_poly = poly.subs(replacement_map)

    return Poly(replaced_poly, *replacement_map.values())

"""
END: WIRING PREDICATES
"""


