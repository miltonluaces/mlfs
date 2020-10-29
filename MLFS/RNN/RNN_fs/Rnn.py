from datetime import datetime
import numpy as np
import sys
from Layers import Layer
from Output import Softmax


class RNN:

    def __init__(self, wordDim, nHid=100, bpttTrunc=4):
        self.wordDim = wordDim
        self.nHid = nHid
        self.bpttTrunc = bpttTrunc
        self.U = np.random.uniform(-np.sqrt(1. / wordDim), np.sqrt(1. / wordDim), (nHid, wordDim))
        self.W = np.random.uniform(-np.sqrt(1. / nHid), np.sqrt(1. / nHid), (nHid, nHid))
        self.V = np.random.uniform(-np.sqrt(1. / nHid), np.sqrt(1. / nHid), (wordDim, nHid))

    # Forward propagation (predicting word probabilities) i.e. x = [0, 179, 341, 416], then its y = [179, 341, 416, 1]
    def Forward(self, x):
        T = len(x) # total number of time steps
        layers = []
        prev = np.zeros(self.nHid)
        for t in range(T):
            layer = Layer()
            input = np.zeros(self.wordDim)
            input[x[t]] = 1
            layer.forward(input, prev, self.U, self.W, self.V)
            prev = layer.s
            layers.append(layer)
        return layers

    def Predict(self, x):
        output = Softmax()
        layers = self.Forward(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def CalcLoss(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.Forward(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulv, y[i])
        return loss / float(len(y))

    def CalcTotalLoss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.CalcLoss(X[i], Y[i])
        return loss / float(len(Y))

    def Bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.Forward(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layers)
        prevT = np.zeros(self.nHid)
        diff = np.zeros(self.nHid)
        for t in range(0, T):
            dmulv = output.diff(layers[t].mulv, y[t])
            input = np.zeros(self.wordDim)
            input[x[t]] = 1
            dprev, dU_t, dW_t, dV_t = layers[t].backward(input, prevT, self.U, self.W, self.V, diff, dmulv)
            prevT = layers[t].s
            dmulv = np.zeros(self.wordDim)
            for i in range(t-1, max(-1, t-self.bpttTrunc-1), -1):
                input = np.zeros(self.wordDim)
                input[x[i]] = 1
                prevI = np.zeros(self.nHid) if i == 0 else layers[i-1].s
                dprev, dU_i, dW_i, dV_i = layers[i].backward(input, prevI, self.U, self.W, self.V, dprev, dmulv)
                dU_t += dU_i
                dW_t += dW_i
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU, dW, dV)

    def SgdStep(self, x, y, m):
        dU, dW, dV = self.Bptt(x, y)
        self.U -= m * dU
        self.V -= m * dV
        self.W -= m * dW

    def Train(self, X, Y, m=0.005, epochs=100, evalLoss=5):
        nExamples = 0
        losses = []
        for epoch in range(epochs):
            if (epoch % evalLoss == 0):
                loss = self.CalcTotalLoss(X, Y)
                losses.append((nExamples, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after n examples = %d epoch=%d: %f" % (time, nExamples, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    m = m * 0.5
                    print("Setting learning rate to %f" % m)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                self.SgdStep(X[i], Y[i], m)
                nExamples += 1
        return losses