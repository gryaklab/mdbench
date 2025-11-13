
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'symbolicregression'))
from typing import List
import requests
import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import sympy as sp

import symbolicregression
from symbolicregression.model import SymbolicTransformerRegressor


def load_model():
    model_path = "model.pt"
    try:
        if not os.path.isfile(model_path):
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)
            model = model.cuda()
        return model

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)

class Estimator:
    def __init__(self, **kwargs):
        self.base_model = load_model()
        self.symbol_names = kwargs.get('symbol_names', None)
        self.target_names = kwargs.get('target_names', None)
        self.system_dim = None
        self.models = [] # list of models for each target

    def fit(self, t_train, X_train, Y_train):
        self.system_dim = Y_train.shape[-1]
        for i in range(self.system_dim):
            model = SymbolicTransformerRegressor(
                        model=self.base_model,
                        max_input_points=200,
                        n_trees_to_refine=100,
                        rescale=True)
            sample_size = 10_000
            if sample_size < len(X_train):
                indices = np.random.choice(len(X_train), size=sample_size, replace=False)
                X_train = X_train[indices]
                Y_train = Y_train[indices]
            model.fit(X_train, Y_train[:, i])
            self.models.append(model)

    def predict(self, t_test, X_test):
        assert self.models is not None, "Model is not trained yet"
        predictions = []
        for i in range(self.system_dim):
            predictions.append(self.models[i].predict(X_test, refinement_type='BFGS'))
        return np.array(predictions)

    def complexity(self) -> int:
        assert self.models is not None, "Model is not trained yet"
        equations = self.to_sympy()
        complexity = 0
        for equation in equations:
            for arg in sp.preorder_traversal(equation.rhs):
                complexity += 1
        return complexity

    def to_str(self) -> str:
        assert self.models is not None, "Model is not trained yet"
        equations_str = []
        equations = self.to_sympy()
        for i in range(self.system_dim):
            rhs = str(equations[i].rhs)
            for j in range(len(self.symbol_names)):
                rhs = rhs.replace(f'x_{j}', self.symbol_names[j])
            equations_str.append(f'{self.target_names[i]}_t = {rhs}')
        return '\n'.join(equations_str)

    def to_sympy(self) -> List[sp.Eq]:
        assert self.models is not None, "Model is not trained yet"
        equations = []
        for i in range(self.system_dim):
            tree = self.models[i].retrieve_tree(refinement_type='BFGS', with_infos=True)["relabed_predicted_tree"]
            rhs = self.models[i].model.env.simplifier.tree_to_sympy_expr(tree)
            lhs = sp.Symbol(f'{self.target_names[i]}_t')
            equations.append(sp.Eq(lhs, rhs))
        return equations
