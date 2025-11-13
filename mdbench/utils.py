import re
from typing import List

import sympy as sp

EXCLUDED_DATASETS = {
    'deepmod': [
        'reaction_diffusion_2d', # The library does not support systems with 2 spatial dimensions and more than 1 variable
        'navier_stokes_channel', # The library does not support systems with 2 spatial dimensions and more than 1 variable
        'navier_stokes_cylinder', # The library does not support systems with 2 spatial dimensions and more than 1 variable
        'heat_soil_uniform_3d_p1', # The library does not support systems with 3d spatial dimension
        'heat_laser', # The library does not support systems with 3d spatial dimension
    ],
    'wsindy': [
        'heat_soil_uniform_3d_p1',
        'navier_stokes_channel',
        'heat_laser',
    ],
    'ewsindy': [
        'heat_soil_uniform_3d_p1',
        'navier_stokes_channel',
        'heat_laser',
    ],
    'deepmod': [
        'heat_laser',
        'heat_soil_uniform_3d_p1',
        'navier_stokes_channel',
        'heat_laser',
    ],
}

def format_equation(expr: str) -> str:
    '''https://github.com/sdascoli/odeformer/blob/main/odeformer/baselines/sindy_wrapper.py#L73C5-L84C20'''
    # <coef> <space> 1 -> <coef> * 1
    expr = re.sub(r"(\d+\.?\d*) (1)", repl=r"\1 * \2", string=expr)
    # <coef> <space> <var> -> <coef> * <var>
    expr = re.sub(r"(\d+\.?\d*) (\w+)", repl=r"\1 * \2", string=expr)
    # <var> <space> <var> -> <coef> * <var>
    expr = re.sub(r"(x_\d+) (x_\d+)", repl=r"\1 * \2", string=expr)
    # <var>^<coef> <space> <var> -> <var>**<coef> * <var>
    expr = re.sub(r"(\w+)\^(\d+)(\w+)", repl=r"\1**\2 * \3", string=expr)
    # python power symbol
    expr = expr.replace("^", "**")
    # sin(u)u -> sin(u) * u OR u_1u_2 -> u_1*u_2 OR u_1u_2_x -> u_1*u_2
    expr =re.sub(r"(?<=[0-9a-zA-Z\)])u", "*u", expr)
    return expr

def str_to_sympy(expressions: str) -> List[sp.Expr]:
    expressions = expressions.split('\n')
    equations = []
    for expr in expressions:
        expr = format_equation(expr)
        if '=' not in expr:
            equation = sp.sympify(expr)
        else:
            lhs, rhs = expr.split('=')
            lhs = sp.sympify(lhs.strip())
            try:
                rhs = sp.sympify(rhs.strip())
            except Exception as e:
                print('Warning: Failed to convert the equation to sympy: ', expr)
                print(e)
                rhs = None
            equation = sp.Eq(lhs, rhs)
        equations.append(equation)
    return equations

def is_excluded_dataset(dataset: str, method: str) -> bool:
    return method in EXCLUDED_DATASETS and dataset in EXCLUDED_DATASETS[method]
