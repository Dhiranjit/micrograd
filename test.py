from engine import Value
from nn import Neuron, Layer, MLP

### Defining an Neuron

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets

n = MLP(3, [3,3,1])




for k in range(1000):
  y_pred = [n(x) for x in xs]
  loss = sum((y - y_p)**2 for y, y_p in zip(ys, y_pred))

  for p in n.parameters():
    p.grad = 0.0
  loss.backward()


  for p in n.parameters():
    p.data += -0.1 * p.grad

  print(k, loss.data)

print(n(xs[1]))