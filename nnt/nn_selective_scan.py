import torch
from einops import rearrange

def isomorphic_selective_scan(
    u,
    delta,
    A,
    B_nn, c_B,
    C_nn, c_C,
    D_pos=None, D_neg_abs=None, c_D=None,
    abs_amin=None, a_max=None,
    return_last_state=False,
):
    """
    SSM with isomorphic B, C, and D decomposition for inputs u in [a_min, a_max].

    B = B_nn - c_B,  C = C_nn - c_C  (c_B, c_C >= 0 constants)

    Dual hidden state:
        h'_t      = deltaA * h'_{t-1}      + deltaB_nn_t * u_t    [non-neg increments]
        h_delta_t = deltaA * h_delta_{t-1} + delta_t * c_B * u_t  [non-neg increments]

    Four non-negative output terms:
        y'  = C_nn @ h'          y1 = C_nn @ h_delta
        y2  = c_C  @ h'          y3 = c_C  @ h_delta

    Final: y = y' - (y1 + y2 - y3)  ==  C @ h  (mathematically identical to original)

    Isomorphic D skip (u in [a_min, a_max]):
        D * u = D_pos * (u + |a_min|) + D_neg_abs * (a_max - u) - c_D
        c_D   = D_pos * |a_min| + D_neg_abs * a_max

    Shapes:
    u: [B,D,L]
    delta: [B,D,L]
    A: [D,N]
    B_nn: [B,N,L]
    C_nn: [B,N,L]
    c_B: [N,]
    c_C: [N,]
    D_pos: [D,] (optional)
    D_neg_abs: [D,] (optional)
    """
    dtype_in = u.dtype
    u     = u.float()
    delta = delta.float()
    A     = A.float()
    B_nn  = B_nn.float()
    C_nn  = C_nn.float()
    c_B   = c_B.float()
    c_C   = c_C.float()

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    h_nn = A.new_zeros(batch, dim, dstate)   # (B, D, N)
    h_delta = A.new_zeros(batch, dim, dstate)   # (B, D, N)

    # Discretization
    deltaA     = torch.exp(-torch.einsum('bdl,dn->bdln', delta, A))
    deltaBu_nn = torch.einsum('bdl,bnl,bdl->bdln', delta, B_nn, u)
    
    ys = []
    last_state = None
    for i in range(seqlen):
        # Update non-negative state: h' = deltaA * h' + deltaB_nn_t
        h_nn = deltaA[: ,:, i] * h_nn + deltaBu_nn[:, :, i]

        # Update correction state: h_delta = deltaA * h_delta + delta_t * c_B * u_t
        h_delta = deltaA[: ,:, i] * h_delta + delta[:, :, i, None] * c_B[None, None, :] * u[:, :, i, None]

        # Four non-negative output terms
        C_t  = C_nn[:, :, i]     # (B, N)
        y_nn = torch.einsum('bn,bdn->bd', C_t, h_nn)
        y1   = torch.einsum('bn,bdn->bd', C_t, h_delta)
        y2   = torch.einsum('n,bdn->bd',  c_C, h_nn)
        y3   = torch.einsum('n,bdn->bd',  c_C, h_delta)
        ys.append(y_nn - (y1 + y2 - y3))

        if i == seqlen - 1:
            last_state = (h_nn - h_delta).clone()

    y = torch.stack(ys, dim=2)   # (B, D, L)

    # Isomorphic D skip: D*u = D_pos*(u+|amin|) + D_neg_abs*(amax-u) - c_D
    if D_pos is not None and D_neg_abs is not None and c_D is not None:
        D_p = rearrange(D_pos.float(),     'd -> d 1')
        D_n = rearrange(D_neg_abs.float(), 'd -> d 1')
        c_d = rearrange(c_D.float(),       'd -> d 1')
        out = y + D_p * (u + abs_amin) + D_n * (a_max - u) - c_d
    else:
        out = y

    out = out.to(dtype_in)
    return out if not return_last_state else (out, last_state)