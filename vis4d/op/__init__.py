"""Vis4D operators.

This is where most of the library APIs are implemented.
All the operators are functors. They are native PyTorch modules and only has a
forward member for functoin invoktions. We follow the princeiple of functional
programming. The operators don't keep internal states besides the operator
weights. The operator computation and call has no side effects.
"""
