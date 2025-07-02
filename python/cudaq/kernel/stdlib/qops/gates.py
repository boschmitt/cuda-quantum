# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir.dialects import quake
from .._base import Error
from .._core.control_flow import for_loop
from .._core.scalars import f64_coerce
from ..collections.veq import Veq
from ..qref import QRef


class Gate:
    __slots__ = ('op', 'num_parameters', 'num_targets', 'is_adjoint')

    def __init__(self,
                 name,
                 *,
                 num_parameters=0,
                 num_targets=1,
                 is_adjoint=False):
        self.op = getattr(quake, '{}Op'.format(name.title()))
        self.num_parameters = num_parameters
        self.num_targets = num_targets
        self.is_adjoint = is_adjoint

    def __call__(self, args, *, is_adjoint: bool = False):
        targets = args[self.num_parameters:]
        parameters = args[:self.num_parameters]
        parameters = [f64_coerce(param) for param in parameters]
        is_adjoint = is_adjoint ^ self.is_adjoint
        if self.num_targets == 1:
            for target in targets:
                if isinstance(target, Veq):
                    length = target.get_length()
                    for_loop(
                        length,
                        lambda i: self.op([],
                                          parameters, [], [target.get_item(i)],
                                          is_adj=is_adjoint))
                elif isinstance(target, QRef):
                    self.op([], parameters, [], [target], is_adj=is_adjoint)
                else:
                    return Error(
                        f'quantum operation {self.op.__name__} on incorrect quantum type {target.type}.'
                    )
        else:
            assert len(targets) == self.num_targets
            self.op([], parameters, [], targets, is_adj=is_adjoint)

    def adj(self, args):
        return self(args, is_adjoint=True)

    def ctrl(self, args):
        targets = args[-self.num_targets:]
        parameters = args[:self.num_parameters]
        parameters = [f64_coerce(param) for param in parameters]
        controls = args[self.num_parameters:-self.num_targets]
        self.op([], parameters, controls, targets, is_adj=self.is_adjoint)


# One-target gates
h = Gate('h', num_targets=1)
s = Gate('s', num_targets=1)
sdg = Gate('s', num_targets=1, is_adjoint=True)
t = Gate('t', num_targets=1)
tdg = Gate('t', num_targets=1, is_adjoint=True)
x = Gate('x', num_targets=1)
y = Gate('y', num_targets=1)
z = Gate('z', num_targets=1)

# One-target gates with parameters
r1 = Gate('r1', num_parameters=1, num_targets=1)
rx = Gate('rx', num_parameters=1, num_targets=1)
ry = Gate('ry', num_parameters=1, num_targets=1)
rz = Gate('rz', num_parameters=1, num_targets=1)
u3 = Gate('u3', num_parameters=3, num_targets=1)

# Two-target gates
swap = Gate('swap', num_targets=2)
