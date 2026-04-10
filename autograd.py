"""
autograd.py — Reverse-mode automatic differentiation engine
No PyTorch, no TensorFlow. Pure NumPy.
Every gradient computed by hand through the computational graph.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable, Union

# Convenient alias so signatures read as intent, not implementation detail.
NDArray = np.ndarray
Scalar = Union[int, float]

class Tensor:
    """
    A multi-dimensional array that tracks its own gradient history.
    Wraps NumPy arrays with automatic differentiation.
    """
    
    def __init__(self, data: Union[NDArray, 'Tensor', Scalar, list],
                 requires_grad: bool = False, name: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data: NDArray = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[NDArray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: List['Tensor'] = []
        self.name = name
    
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def T(self): return self.transpose()
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def zero_grad(self):
        self.grad = None
    
    def backward(self, grad: Optional[NDArray] = None) -> None:
        if grad is None:
            assert self.data.size == 1, "backward() without grad only for scalars"
            grad = np.ones_like(self.data)
        
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = grad
        for v in reversed(topo):
            v._backward()
    
    def _init_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
    
    # ─── Arithmetic ────────────────────────────────────────────────────────────
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                g = out.grad
                # Sum over broadcasted dims
                self.grad += _unbroadcast(g, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad, other.shape)
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1)
    
    def __rsub__(self, other): return (self * -1).__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.shape)
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1
    
    def __pow__(self, exp):
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += exp * (self.data ** (exp - 1)) * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def __neg__(self): return self * -1
    
    # ─── Matrix Operations ─────────────────────────────────────────────────────
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                if out.grad.ndim >= 2:
                    self.grad += out.grad @ other.data.swapaxes(-1, -2)
                else:
                    self.grad += np.outer(out.grad, other.data)
            if other.requires_grad:
                other._init_grad()
                if out.grad.ndim >= 2:
                    other.grad += self.data.swapaxes(-1, -2) @ out.grad
                else:
                    other.grad += np.outer(self.data, out.grad)
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def __matmul__(self, other): return self.matmul(other)
    
    def transpose(self, axes=None):
        if axes is None:
            if self.ndim == 2:
                axes = (1, 0)
            else:
                axes = tuple(range(self.ndim - 1, -1, -1))
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                inv_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inv_axes)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.reshape(self.shape)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def view(self, *shape): return self.reshape(*shape)
    
    # ─── Reductions ────────────────────────────────────────────────────────────
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                self.grad += np.broadcast_to(g, self.shape)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)
    
    # ─── Activations ───────────────────────────────────────────────────────────
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def gelu(self):
        """Gaussian Error Linear Unit — used in modern LLMs"""
        x = self.data
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        out = Tensor(x * cdf, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                t = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
                sech2 = 1.0 - t**2
                dcdf = 0.5 * sech2 * np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x**2)
                self.grad += out.grad * (cdf + x * dcdf)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
        out = Tensor(s, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * s * (1 - s)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def silu(self):
        """SiLU / Swish — x * sigmoid(x), used in SwiGLU"""
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
        out = Tensor(self.data * s, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                s2 = 1.0 / (1.0 + np.exp(-np.clip(self.data, -88, 88)))
                self.grad += out.grad * s2 * (1 + self.data * (1 - s2))
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)  # numerical stability
        e = np.exp(x)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                # d/dx[softmax] = softmax * (I - softmax^T) => simplified:
                g = out.grad
                self.grad += s * (g - (g * s).sum(axis=axis, keepdims=True))
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad / (self.data + 1e-9)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def exp(self):
        e = np.exp(np.clip(self.data, -88, 88))
        out = Tensor(e, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * e
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def sqrt(self):
        s = np.sqrt(np.abs(self.data) + 1e-8)
        out = Tensor(s, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * 0.5 / s
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    # ─── Indexing ──────────────────────────────────────────────────────────────
    
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                np.add.at(self.grad, idx, out.grad)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def concat(self, other, axis=0):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.concatenate([self.data, other.data], axis=axis),
                     requires_grad=self.requires_grad or other.requires_grad)
        split = self.data.shape[axis]
        
        def _backward():
            parts = np.split(out.grad, [split], axis=axis)
            if self.requires_grad:
                self._init_grad()
                self.grad += parts[0]
            if other.requires_grad:
                other._init_grad()
                other.grad += parts[1]
        
        out._backward = _backward
        out._prev = [self, other]
        return out
    
    def expand_dims(self, axis):
        out = Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.squeeze(axis)
        
        out._backward = _backward
        out._prev = [self]
        return out
    
    def item(self):
        return float(self.data.flat[0])


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _unbroadcast(grad: NDArray, shape: tuple) -> NDArray:
    """Sum gradient over broadcasted dimensions to restore original shape."""
    if grad.shape == shape:
        return grad
    # Sum axes that were broadcast
    ndim_diff = grad.ndim - len(shape)
    # Sum leading dims added by broadcast
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    # Sum dims that were size-1 in original
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def zeros(*shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad=requires_grad)

def randn(*shape, requires_grad=False, scale=1.0):
    return Tensor(np.random.randn(*shape).astype(np.float32) * scale, 
                  requires_grad=requires_grad)

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)
