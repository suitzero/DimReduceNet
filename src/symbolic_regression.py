import numpy as np
import scipy.optimize as opt
from dataclasses import dataclass

class Node:
    pass

@dataclass
class Var(Node):
    name: str = 'x'
    def __str__(self): return self.name

@dataclass
class Const(Node):
    idx: int
    def __str__(self): return f"c[{self.idx}]"

@dataclass
class UnaryOp(Node):
    op: str
    child: Node
    def __str__(self): return f"{self.op}({self.child})"

@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node
    def __str__(self): return f"({self.left} {self.op} {self.right})"

def generate_trees_by_size(size, num_consts=1, use_t=False):
    """Generates all abstract expression trees of a given size (number of nodes)."""
    if size == 1:
        yield Var('x')
        if use_t:
            yield Var('t')
        yield Const(0)
    elif size > 1:
        # Unary operations
        for child in generate_trees_by_size(size - 1, num_consts, use_t=use_t):
            for op in ['sin', 'cos', 'exp', 'abs', 'sqrt']:
                yield UnaryOp(op, child)
        
        # Binary operations
        for left_size in range(1, size):
            right_size = size - 1 - left_size
            if right_size < 1:
                continue
            for left in generate_trees_by_size(left_size, num_consts, use_t=use_t):
                for right in generate_trees_by_size(right_size, num_consts, use_t=use_t):
                    for op in ['+', '-', '*', '/', 'union', 'intersection', 'difference']:
                        # Simple pruning for commutativity and redundancy
                        if op in ['+', '*', 'union', 'intersection'] and str(left) > str(right):
                            continue
                        if isinstance(left, Const) and isinstance(right, Const):
                            continue
                        if op == '/' and isinstance(right, Const):
                            continue
                        yield BinOp(op, left, right)

def assign_const_indices(node: Node) -> tuple[Node, int]:
    """Assigns sequential indices to all constants in the tree."""
    def assign(n: Node, idx: int) -> tuple[Node, int]:
        if isinstance(n, Var):
            return Var(n.name), idx
        elif isinstance(n, Const):
            return Const(idx), idx + 1
        elif isinstance(n, UnaryOp):
            new_child, next_idx = assign(n.child, idx)
            return UnaryOp(n.op, new_child), next_idx
        elif isinstance(n, BinOp):
            new_left, next_idx = assign(n.left, idx)
            new_right, next_idx2 = assign(n.right, next_idx)
            return BinOp(n.op, new_left, new_right), next_idx2
        return n, idx
    return assign(node, 0)

def compile_tree(node: Node):
    """Compiles an AST node to an executable Python function (using numpy) and formatting string."""
    node, num_c = assign_const_indices(node)
    
    def build_expr(n: Node) -> str:
        if isinstance(n, Var):
            if n.name == 't':
                return "t"
            return "x"
        elif isinstance(n, Const):
            return f"c[{n.idx}]"
        elif isinstance(n, UnaryOp):
            if n.op == 'sqrt':
                return f"np.sqrt(np.abs({build_expr(n.child)}))"
            return f"np.{n.op}({build_expr(n.child)})"
        elif isinstance(n, BinOp):
            if n.op == '/':
                return f"({build_expr(n.left)} / ({build_expr(n.right)} + 1e-8))"
            elif n.op == 'union':
                return f"np.minimum({build_expr(n.left)}, {build_expr(n.right)})"
            elif n.op == 'intersection':
                return f"np.maximum({build_expr(n.left)}, {build_expr(n.right)})"
            elif n.op == 'difference':
                return f"np.maximum({build_expr(n.left)}, -({build_expr(n.right)}))"
            return f"({build_expr(n.left)} {n.op} {build_expr(n.right)})"
    
    expr_str = build_expr(node)
    
    def build_format(n: Node) -> str:
        if isinstance(n, Var):
            if n.name == 't':
                return "t"
            return "x"
        elif isinstance(n, Const):
            return "{}"
        elif isinstance(n, UnaryOp):
            return f"{n.op}({build_format(n.child)})"
        elif isinstance(n, BinOp):
            if n.op in ['union', 'intersection', 'difference']:
                return f"{n.op}({build_format(n.left)}, {build_format(n.right)})"
            return f"({build_format(n.left)} {n.op} {build_format(n.right)})"
            
    format_str = build_format(node)
    
    func_code = f"def f(x, t, c):\n    return {expr_str}"
    local_vars = {'np': np}
    exec(func_code, local_vars)
    
    return local_vars['f'], num_c, format_str

def format_eq(format_str: str, constants: list) -> str:
    """Formats the expression string with rounded optimized constants."""
    s = format_str
    for c in constants:
        c_round = round(c, 3)
        if c_round == int(c_round):
            c_str = str(int(c_round))
        else:
            c_str = str(c_round)
        s = s.replace('{}', c_str, 1)
    
    # Simplification passes for string representation
    s = s.replace('+ -', '- ')
    
    return s

class SymbolicRegressionEngine:
    def __init__(self, max_size=8, threshold=1e-3, seed=42, use_t=False):
        self.max_size = max_size
        self.threshold = threshold
        self.seed = seed
        self.use_t = use_t
        
    def fit(self, x_data, y_data, t_data=None):
        """
        Fits the shortest possible mathematical function to the data.
        Returns the formatted function string and the MSE.
        """
        if t_data is None:
            t_data = np.zeros_like(x_data)

        best_str = None
        best_mse = float('inf')
        
        for size in range(1, self.max_size + 1):
            for tree in generate_trees_by_size(size, use_t=self.use_t):
                try:
                    f, num_c, format_str = compile_tree(tree)
                except Exception:
                    continue
                    
                if num_c == 0:
                    try:
                        y_pred = f(x_data, t_data, [])
                        if np.isscalar(y_pred):
                            y_pred = np.full_like(x_data, y_pred)
                        mse = np.mean((y_data - y_pred)**2)
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_str = format_str
                            
                        if mse < self.threshold:
                            return best_str, mse
                    except Exception:
                        pass
                    continue
                
                # Optimize constants
                def objective(c):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            y_pred = f(x_data, t_data, c)
                            if np.isscalar(y_pred):
                                y_pred = np.full_like(x_data, y_pred)
                            if not np.all(np.isfinite(y_pred)):
                                return float('inf')
                            return np.mean((y_data - y_pred)**2)
                        except Exception:
                            return float('inf')
                            
                bounds = [(-10.0, 10.0)] * num_c
                res = opt.differential_evolution(
                    objective, bounds, seed=self.seed, popsize=10, maxiter=20, tol=self.threshold
                )
                
                if res.fun < best_mse:
                    best_mse = res.fun
                    best_str = format_eq(format_str, res.x)
                    
                if res.fun < self.threshold:
                    return format_eq(format_str, res.x), res.fun
                    
        return best_str, best_mse
