
import numpy as np
from odeformer.model import SymbolicTransformerRegressor

hyper_params = {
    'beam_size': [50],
    'beam_temperature': [0.05, 0.1, 0.2, 0.3, 0.5],
}

class Estimator:
    def __init__(self, **kwargs):
        self.model = SymbolicTransformerRegressor(from_pretrained=True)
        self.symbol_names = kwargs.get('symbol_names', None)
        self.target_names = kwargs.get('target_names', None)
        self.system_dim = None

        beam_size = kwargs.get('beam_size', 50)
        beam_temperature = kwargs.get('beam_temperature', 0.1)
        model_args = {
            'beam_size':beam_size,
            'beam_temperature':beam_temperature
        }
        self.model.set_model_args(model_args)

    def fit(self, t_train, X_train, Y_train):
        self.system_dim = Y_train.shape[-1]
        self.model.fit(t_train, X_train)

    def predict(self, t_test, X_test):
        assert self.model is not None, "Model is not trained yet"
        return self.model.predictions[0][0].val(X_test, [0])

    def complexity(self):
        '''Return the complexity of the model defined as the number of operations in the equation'''
        assert self.model is not None, "Model is not trained yet"
        tree = self.model.predictions[0][0]
        return len(tree.prefix().replace("|", "").split(","))

    def to_str(self) -> str:
        assert self.model is not None, "Model is not trained yet"
        equations = []
        if self.target_names is None:
            self.target_names = [f'u{i}' for i in range(self.system_dim)]
        expressions = str(self.model.predictions[0][0]).split('|')
        for i in range(len(expressions)):
            lhs = f'{self.target_names[i]}_t'
            rhs = expressions[i]
            equations.append(f'{lhs} = {rhs}')
        return '\n'.join(equations)
