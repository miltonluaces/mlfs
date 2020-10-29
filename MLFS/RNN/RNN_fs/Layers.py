from Activation import Tanh
from Gates import AddGate, MultiplyGate

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class Layer:

    def forward(self, x, prev, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev, U, W, V, diff, dmulv):
        self.forward(x, prev, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev = mulGate.backward(W, prev, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev, dU, dW, dV)