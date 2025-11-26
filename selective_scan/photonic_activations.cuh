/******************************************************************************
 * Photonic Activation Functions for CUDA
 * 
 * These are device functions implementing photonic-compatible activations
 * that can replace standard activations (softplus, silu) in the Mamba
 * selective scan kernels.
 ******************************************************************************/

#pragma once

#include <cuda_fp16.h>
#ifndef USE_ROCM
    #include <cuda_bf16.h>
#else
    #include <hip/hip_bf16.h>
#endif

// ============================================================================
// Activation Type Enum - passed via params to select activation at runtime
// ============================================================================
enum class PhotonicActivationType : int {
    STANDARD = 0,       // Original softplus/silu
    PSIGMOID = 1,
    PSINUSOIDAL = 2,
    PTANHLIKE = 3,
    PELULIKE = 4,
    PINVELU = 5,
    PDSINSQ = 6,
    PRESIN = 7,
    PEXPSIN = 8
};

// ============================================================================
// PSigmoid: a2 + (a1 - a2) / (1 + exp((x - x0) / d))
// Default params: a1=0.0198, a2=0.07938, x0=1.26092, d=0.48815
// ============================================================================
struct PSigmoidParams {
    float a1 = 0.0198f;
    float a2 = 0.07938f;
    float x0 = 1.26092f;
    float d = 0.48815f;
};

__device__ __forceinline__ float psigmoid_fwd(float x, const PSigmoidParams& p) {
    float z = (x - p.x0) / p.d;
    z = fminf(fmaxf(z, -80.0f), 80.0f);  // Prevent overflow
    return p.a2 + (p.a1 - p.a2) / (1.0f + expf(z));
}

__device__ __forceinline__ float psigmoid_bwd(float x, float grad_out, const PSigmoidParams& p) {
    float z = (x - p.x0) / p.d;
    z = fminf(fmaxf(z, -80.0f), 80.0f);
    float exp_z = expf(z);
    float denom = 1.0f + exp_z;
    float sigmoid_deriv = exp_z / (denom * denom);
    return grad_out * (p.a1 - p.a2) * (-1.0f / p.d) * sigmoid_deriv;
}

// ============================================================================
// PSinusoidal: sin^2(pi/2 * x) for x in (0,1), else 0 or 1
// ============================================================================
__device__ __forceinline__ float psinusoidal_fwd(float x) {
    if (x > 0.0f && x < 1.0f) {
        float s = sinf(1.5707963267948966f * x);  // pi/2
        return s * s;
    }
    return (x >= 1.0f) ? 1.0f : 0.0f;
}

__device__ __forceinline__ float psinusoidal_bwd(float x, float grad_out) {
    if (x > 0.0f && x < 1.0f) {
        // d/dx[sin^2(pi/2 * x)] = 2 * sin(pi/2 * x) * cos(pi/2 * x) * pi/2
        //                       = pi/2 * sin(pi * x)
        return grad_out * 1.5707963267948966f * sinf(3.141592653589793f * x);
    }
    return 0.0f;
}

// ============================================================================
// PTanhLike: a + (d + b*sinh(x - x0)) / (e + c*cosh(x - x0))
// Default params: a=0.24057, b=0.34184, c=1.74544, d=-1.65912, e=3.2698, x0=-0.30873
// ============================================================================
struct PTanhLikeParams {
    float a = 0.24057f;
    float b = 0.34184f;
    float c = 1.74544f;
    float d = -1.65912f;
    float e = 3.2698f;
    float x0 = -0.30873f;
};

__device__ __forceinline__ float ptanhlike_fwd(float x, const PTanhLikeParams& p) {
    float t = x - p.x0;
    float sinh_t = sinhf(t);
    float cosh_t = coshf(t);
    return p.a + (p.d + p.b * sinh_t) / (p.e + p.c * cosh_t);
}

__device__ __forceinline__ float ptanhlike_bwd(float x, float grad_out, const PTanhLikeParams& p) {
    float t = x - p.x0;
    float sinh_t = sinhf(t);
    float cosh_t = coshf(t);
    float num = p.d + p.b * sinh_t;
    float denom = p.e + p.c * cosh_t;
    // Quotient rule: d/dx[num/denom] = (num' * denom - num * denom') / denom^2
    // num' = b * cosh(t), denom' = c * sinh(t)
    float num_deriv = p.b * cosh_t;
    float denom_deriv = p.c * sinh_t;
    float deriv = (num_deriv * denom - num * denom_deriv) / (denom * denom);
    return grad_out * deriv;
}

// ============================================================================
// PELULike: For x >= x0: b*(x - x0) + c, else: a*(exp(x - x0) - 1) + c
// Default params: a=0.0368, b=0.18175, c=-0.01957, x0=0.37042
// ============================================================================
struct PELULikeParams {
    float a = 0.0368f;
    float b = 0.18175f;
    float c = -0.01957f;
    float x0 = 0.37042f;
};

__device__ __forceinline__ float pelulike_fwd(float x, const PELULikeParams& p) {
    if (x >= p.x0) {
        return p.b * (x - p.x0) + p.c;
    } else {
        return p.a * (expf(x - p.x0) - 1.0f) + p.c;
    }
}

__device__ __forceinline__ float pelulike_bwd(float x, float grad_out, const PELULikeParams& p) {
    if (x >= p.x0) {
        return grad_out * p.b;
    } else {
        return grad_out * p.a * expf(x - p.x0);
    }
}

// ============================================================================
// PInvELU: For x <= x0: b*(x - x0) + c, else: a/(exp(x0 - x) + 1) + d
// Default params: a=0.02395, b=0.15568, c=0.08616, d=0.04855, x0=-0.2
// ============================================================================
struct PInvELUParams {
    float a = 0.02395f;
    float b = 0.15568f;
    float c = 0.08616f;
    float d = 0.04855f;
    float x0 = -0.2f;
};

__device__ __forceinline__ float pinvelu_fwd(float x, const PInvELUParams& p) {
    if (x <= p.x0) {
        return p.b * (x - p.x0) + p.c;
    } else {
        return p.a / (expf(p.x0 - x) + 1.0f) + p.d;
    }
}

__device__ __forceinline__ float pinvelu_bwd(float x, float grad_out, const PInvELUParams& p) {
    if (x <= p.x0) {
        return grad_out * p.b;
    } else {
        float exp_val = expf(p.x0 - x);
        float denom = exp_val + 1.0f;
        // d/dx[a / (exp(x0-x) + 1)] = a * exp(x0-x) / (exp(x0-x) + 1)^2
        return grad_out * p.a * exp_val / (denom * denom);
    }
}

// ============================================================================
// PDSinSq: Asymmetric sine-squared
// For x >= 0: a*sin^2(d*(x + x0)) + c
// For x < 0:  b*sin^2(e*(-x + x0)) + c
// Default params: a=1.7917, b=0.8571, c=0.2514, d=1.1066, e=0.1416, x0=0.9807
// ============================================================================
struct PDSinSqParams {
    float a = 1.7917f;
    float b = 0.8571f;
    float c = 0.2514f;
    float d = 1.1066f;
    float e = 0.1416f;
    float x0 = 0.9807f;
};

__device__ __forceinline__ float pdsinsq_fwd(float x, const PDSinSqParams& p) {
    x = fminf(fmaxf(x, -1.9f), 1.9f);  // Clamp
    if (x >= 0.0f) {
        float s = sinf(p.d * (x + p.x0));
        return p.a * s * s + p.c;
    } else {
        float s = sinf(p.e * (-x + p.x0));
        return p.b * s * s + p.c;
    }
}

__device__ __forceinline__ float pdsinsq_bwd(float x, float grad_out, const PDSinSqParams& p) {
    x = fminf(fmaxf(x, -1.9f), 1.9f);
    if (x >= 0.0f) {
        // d/dx[a*sin^2(d*(x+x0))] = 2*a*d*sin(d*(x+x0))*cos(d*(x+x0))
        //                        = a*d*sin(2*d*(x+x0))
        float arg = p.d * (x + p.x0);
        return grad_out * p.a * p.d * sinf(2.0f * arg);
    } else {
        // d/dx[b*sin^2(e*(-x+x0))] = 2*b*sin(e*(-x+x0))*cos(e*(-x+x0))*e*(-1)
        //                         = -b*e*sin(2*e*(-x+x0))
        float arg = p.e * (-x + p.x0);
        return grad_out * (-p.b * p.e * sinf(2.0f * arg));
    }
}

// ============================================================================
// PReSin: For x >= x0: a*sin^2(d*(x - x0)) + c, else: a*(x - x0) + c
// Default params: a=0.23299, b=0.00047, c=0.01692, d=-0.71482, x0=0.44184
// ============================================================================
struct PReSinParams {
    float a = 0.23299f;
    float b = 0.00047f;  // Not used in main formula
    float c = 0.01692f;
    float d = -0.71482f;
    float x0 = 0.44184f;
};

__device__ __forceinline__ float presin_fwd(float x, const PReSinParams& p) {
    if (x >= p.x0) {
        float s = sinf(p.d * (x - p.x0));
        return p.a * s * s + p.c;
    } else {
        return p.a * (x - p.x0) + p.c;
    }
}

__device__ __forceinline__ float presin_bwd(float x, float grad_out, const PReSinParams& p) {
    if (x >= p.x0) {
        float arg = p.d * (x - p.x0);
        return grad_out * p.a * p.d * sinf(2.0f * arg);
    } else {
        return grad_out * p.a;
    }
}

// ============================================================================
// PExpSin: For x >= x0: a*sin^2(d*(x - x0)) + c, else: b*exp(x - 1) + c
// Default params: a=1, b=1, c=0, d=1, x0=0
// ============================================================================
struct PExpSinParams {
    float a = 1.0f;
    float b = 1.0f;
    float c = 0.0f;
    float d = 1.0f;
    float x0 = 0.0f;
};

__device__ __forceinline__ float pexpsin_fwd(float x, const PExpSinParams& p) {
    if (x >= p.x0) {
        float s = sinf(p.d * (x - p.x0));
        return p.a * s * s + p.c;
    } else {
        return p.b * expf(x - 1.0f) + p.c;
    }
}

__device__ __forceinline__ float pexpsin_bwd(float x, float grad_out, const PExpSinParams& p) {
    if (x >= p.x0) {
        float arg = p.d * (x - p.x0);
        return grad_out * p.a * p.d * sinf(2.0f * arg);
    } else {
        return grad_out * p.b * expf(x - 1.0f);
    }
}


// ============================================================================
// Unified Photonic Activation Interface
// These functions dispatch based on activation type
// ============================================================================

// For delta activation (replacing softplus)
__device__ __forceinline__ float photonic_delta_activation_fwd(
    float x, 
    PhotonicActivationType act_type,
    bool use_standard_softplus = true
) {
    switch (act_type) {
        case PhotonicActivationType::STANDARD:
            // Standard softplus: log(1 + exp(x))
            return (use_standard_softplus && x <= 20.0f) ? log1pf(expf(x)) : x;
        case PhotonicActivationType::PSIGMOID:
            return psigmoid_fwd(x, PSigmoidParams{});
        case PhotonicActivationType::PSINUSOIDAL:
            return psinusoidal_fwd(x);
        case PhotonicActivationType::PTANHLIKE:
            return ptanhlike_fwd(x, PTanhLikeParams{});
        case PhotonicActivationType::PELULIKE:
            return pelulike_fwd(x, PELULikeParams{});
        case PhotonicActivationType::PINVELU:
            return pinvelu_fwd(x, PInvELUParams{});
        case PhotonicActivationType::PDSINSQ:
            return pdsinsq_fwd(x, PDSinSqParams{});
        case PhotonicActivationType::PRESIN:
            return presin_fwd(x, PReSinParams{});
        case PhotonicActivationType::PEXPSIN:
            return pexpsin_fwd(x, PExpSinParams{});
        default:
            return x;  // Identity
    }
}

__device__ __forceinline__ float photonic_delta_activation_bwd(
    float x, 
    float grad_out,
    PhotonicActivationType act_type,
    bool use_standard_softplus = true
) {
    switch (act_type) {
        case PhotonicActivationType::STANDARD:
            // Softplus derivative: sigmoid(x) = 1 / (1 + exp(-x))
            if (use_standard_softplus && x <= 20.0f) {
                float exp_neg_x = expf(-x);
                return grad_out / (1.0f + exp_neg_x);
            }
            return grad_out;
        case PhotonicActivationType::PSIGMOID:
            return psigmoid_bwd(x, grad_out, PSigmoidParams{});
        case PhotonicActivationType::PSINUSOIDAL:
            return psinusoidal_bwd(x, grad_out);
        case PhotonicActivationType::PTANHLIKE:
            return ptanhlike_bwd(x, grad_out, PTanhLikeParams{});
        case PhotonicActivationType::PELULIKE:
            return pelulike_bwd(x, grad_out, PELULikeParams{});
        case PhotonicActivationType::PINVELU:
            return pinvelu_bwd(x, grad_out, PInvELUParams{});
        case PhotonicActivationType::PDSINSQ:
            return pdsinsq_bwd(x, grad_out, PDSinSqParams{});
        case PhotonicActivationType::PRESIN:
            return presin_bwd(x, grad_out, PReSinParams{});
        case PhotonicActivationType::PEXPSIN:
            return pexpsin_bwd(x, grad_out, PExpSinParams{});
        default:
            return grad_out;
    }
}

// For gating activation (replacing silu)
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float photonic_gate_activation_fwd(
    float x,
    PhotonicActivationType act_type
) {
    switch (act_type) {
        case PhotonicActivationType::STANDARD: {
            // Standard SiLU
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            return x * sigmoid_x;
        }
        case PhotonicActivationType::PSIGMOID:
            return psigmoid_fwd(x, PSigmoidParams{});
        case PhotonicActivationType::PSINUSOIDAL:
            return psinusoidal_fwd(x);
        case PhotonicActivationType::PTANHLIKE:
            return ptanhlike_fwd(x, PTanhLikeParams{});
        case PhotonicActivationType::PELULIKE:
            return pelulike_fwd(x, PELULikeParams{});
        case PhotonicActivationType::PINVELU:
            return pinvelu_fwd(x, PInvELUParams{});
        case PhotonicActivationType::PDSINSQ:
            return pdsinsq_fwd(x, PDSinSqParams{});
        case PhotonicActivationType::PRESIN:
            return presin_fwd(x, PReSinParams{});
        case PhotonicActivationType::PEXPSIN:
            return pexpsin_fwd(x, PExpSinParams{});
        default:
            return x;
    }
}

// For gating backward - note: for silu, we need both the activation value and gradient
__device__ __forceinline__ void photonic_gate_activation_bwd(
    float z_val,           // Input z
    float out_val,         // SSM output (before gating)
    float dout_val,        // Gradient of loss w.r.t. final output
    PhotonicActivationType act_type,
    float& dz_out,         // Output: gradient w.r.t. z
    float& dout_vals     // Output: gradient to pass to SSM backward
) {
    switch (act_type) {
        case PhotonicActivationType::STANDARD: {
            // Standard SiLU backward
            float z_sigmoid = 1.0f / (1.0f + expf(-z_val));
            float z_silu = z_val * z_sigmoid;
            // d(out * silu(z))/dz = out * silu'(z) = out * sigmoid(z) * (1 + z*(1-sigmoid(z)))
            dz_out = dout_val * out_val * z_sigmoid * (1.0f + z_val * (1.0f - z_sigmoid));
            // d(out * silu(z))/d(out) = silu(z)
            dout_vals = dout_val * z_silu;
            break;
        }
        case PhotonicActivationType::PSIGMOID: {
            PSigmoidParams p;
            float act_val = psigmoid_fwd(z_val, p);
            dz_out = dout_val * out_val * psigmoid_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PSINUSOIDAL: {
            float act_val = psinusoidal_fwd(z_val);
            dz_out = dout_val * out_val * psinusoidal_bwd(z_val, 1.0f);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PTANHLIKE: {
            PTanhLikeParams p;
            float act_val = ptanhlike_fwd(z_val, p);
            dz_out = dout_val * out_val * ptanhlike_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PELULIKE: {
            PELULikeParams p;
            float act_val = pelulike_fwd(z_val, p);
            dz_out = dout_val * out_val * pelulike_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PINVELU: {
            PInvELUParams p;
            float act_val = pinvelu_fwd(z_val, p);
            dz_out = dout_val * out_val * pinvelu_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PDSINSQ: {
            PDSinSqParams p;
            float act_val = pdsinsq_fwd(z_val, p);
            dz_out = dout_val * out_val * pdsinsq_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PRESIN: {
            PReSinParams p;
            float act_val = presin_fwd(z_val, p);
            dz_out = dout_val * out_val * presin_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        case PhotonicActivationType::PEXPSIN: {
            PExpSinParams p;
            float act_val = pexpsin_fwd(z_val, p);
            dz_out = dout_val * out_val * pexpsin_bwd(z_val, 1.0f, p);
            dout_vals = dout_val * act_val;
            break;
        }
        default:
            dz_out = 0.0f;
            dout_vals = dout_val;
    }
}
