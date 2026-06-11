import torch as T


class H_ABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1.0, lr_out=1.0, lr=1.0, g=1.0):
        p.add_((self.m_abs(p, grad, lr_in).mul(-lr_out).mul(g) +
                grad.mul(-lr).mul(1.0 - g)))

    @T.no_grad()
    def m_abs(self, p, grad, lr_in):
        return p.abs().mul(grad.mul(lr_in).tanh())

    @T.no_grad()
    def tanh_part(self, grad, lr_in=1):
        return grad.mul(lr_in).tanh()

    @T.no_grad()
    def update(self, p, grad, lr_out=1, lr=1.0, g=1.0):
        p.add_(p.abs().mul(grad).mul(lr_out).mul(g) + grad.mul(lr).mul(1 - g))

    def __repr__(self):
        return "H_ABS"


class M_ABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1, lr_out=1, g=None):
        p.addcmul_(p.abs(), grad.mul(lr_in).tanh(), value=-lr_out)

    @T.no_grad()
    def tanh_part(self, grad, lr_in=1):
        return grad.mul(lr_in).tanh()

    @T.no_grad()
    def update(self, p, grad, lr_out=1):
        p.addcmul_(p.abs(), grad, value=-lr_out)

    def __repr__(self):
        return "M_ABS"


class M_SPOW(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1, lr_out=1, g=None):
        p.mul_(T.pow(2, grad.mul(lr_in).tanh().mul(lr_out).mul(-p.sign())))

    @T.no_grad()
    def tanh_part(self, grad, lr_in=1):
        return grad.mul(lr_in).tanh()

    @T.no_grad()
    def update(self, p, grad, lr_out):
        p.mul_(T.pow(2, grad.mul(lr_out).mul(-p.sign())))

    def __repr__(self):
        return "M_SPOW"


class N_Clip(object):
    @T.no_grad()
    def __call__(self, p, grad, lr):
        if p.grad is not None:
            p.add_(grad, alpha=-lr).clamp_min_(0)

    def __repr__(self):
        return "N_CLIP"


class N_ABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr):
        if p.grad is not None:
            p.add_(grad, alpha=-lr).abs_()

    def __repr__(self):
        return "N_ABS"


class U_Add(object):
    """Standard (Euclidean) Adam step: p <- p - lr_out * update.

    The MAdam-compatible additive update (= MAdam's commented-out vanilla-Adam
    line). Used for sign-free parameters (A_log, RMSNorm weights) so they are not
    constrained by the sign-preserving multiplicative updates.
    """
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1.0, lr_out=1.0, g=None):
        p.add_(grad, alpha=-lr_out)

    def __repr__(self):
        return "U_ADD"
