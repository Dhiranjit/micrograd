import random
import math
from engine import Value
from typing import List

class Neuron:
    def __init__(self, n_in: int):
        # Xavier Initialization
        limit = math.sqrt(6 / n_in)
        self.w = [Value(random.uniform(-limit, limit)) for _ in range(n_in)]
        self.b = Value(0.0)

         

    def __call__(self, x: List[float]):
        dot = sum((wi* xi for wi, xi in zip(self.w, x)), self.b)
        out  = dot.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
 

class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x: List[float]):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return[p for nueron in self.neurons for p in nueron.parameters()]


class MLP:
    def __init__(self,n_in: int, n_outs: List[int]):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]
    
    def __call__(self, x: List[float]):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    