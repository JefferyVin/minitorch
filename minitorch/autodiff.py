from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    f_vals = f(*vals)
    # print(f_vals)
    f_vals_grad = f(*[x if idx != arg else x + epsilon for idx, x in enumerate(vals)])
    # print(f_vals_grad)
    return (f_vals_grad - f_vals) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Any) -> Iterable[Any]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    topo = []

    v = set()

    def visit(n: Variable):
        if n.unique_id not in v:
            if not n.is_constant():
                for m in n.parents:
                    visit(m)
            v.add(n.unique_id)
            topo.append(n)

    visit(variable)
    return topo


def backpropagate(variable: Any, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Outer loop: Step 1 - Call topological sort
    topo = topological_sort(variable)

    # Outer loop: Step 2 - Create dict of variables and derivatives
    derivatives = {node.unique_id: 0.0 for node in topo}
    derivatives[variable.unique_id] = deriv

    # Outer loop: Step 3 - For each node in backward order
    for node in reversed(topo):
        node_deriv = derivatives[node.unique_id]
        # print(
        #     f"Processing node: {node}, is_leaf: {node.is_leaf()}, derivative: {derivatives[node.unique_id]}"
        # )

        if node.is_leaf():
            node.accumulate_derivative(node_deriv)
        elif not node.is_constant():
            for var, partial_deriv in node.chain_rule(node_deriv):
                derivatives[var.unique_id] += partial_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
