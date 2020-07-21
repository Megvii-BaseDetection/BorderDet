#include "torch/extension.h"

namespace cvpods {
template <typename scalar_t>
void ComputeAdaptiveLrAfterTypeCheck(
    const scalar_t &param_norm,
    const scalar_t &grad_norm,
    const scalar_t weight_decay,
    const scalar_t eps,
    const scalar_t trust_coef,
    scalar_t *out) {
  // 1. The case that `param_norm` is `zero` means all elements of the parameter 
  // are `zero` (In general, it occurs when right after the parameter initialized 
  // as `zero`).  In this case, `adaptive_lr` will be calculated as `zero`, which 
  // may be the reason for breaking parameter updates.  In this context, we construct 
  // LARS to use only wrapped optimizer's algorithm when this situation occurs by 
  // converting `adaptive_lr` to `one`.
  //
  // 2. The case that `grad_norm` is `zero` means all elements of the gradient are 
  // `zero` (In general, it occurs when backward propagation doesn't work correctly).  
  // In this case, it can be interpreted as there exists an exceptional situation, 
  // which may result in inappropriate parameter updates.  In this context, we 
  // construct LARS to pass the responsibility of handling the exceptional case 
  // to the wrapped optimizer when this exception occurs by converting `adaptive_lr` 
  // to `one`.
  if (param_norm > 0 && grad_norm > 0) {
    scalar_t divisor = grad_norm + weight_decay * param_norm + eps;
    *out = param_norm / divisor * trust_coef;
  } else {
    *out = 1.0;
  }
}

// CUDA function interface
void ComputeAdaptiveLrOnDevice(
    torch::Tensor param_norm,
    torch::Tensor grad_norm,
    double weight_decay,
    double eps,
    double trust_coef,
    torch::Tensor out);

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")

torch::Tensor ComputeAdaptiveLr(
    torch::Tensor param_norm,
    torch::Tensor grad_norm,
    double weight_decay,
    double eps,
    double trust_coef,
    torch::Tensor out) {
  CHECK_CONTIGUOUS(param_norm);
  CHECK_CONTIGUOUS(grad_norm);
  CHECK_CONTIGUOUS(out);

  if (param_norm.type().is_cuda() && grad_norm.type().is_cuda()) {
    ComputeAdaptiveLrOnDevice(
        param_norm,
        grad_norm,
        weight_decay,
        eps,
        trust_coef,
        out);
  } else {
    CHECK_CPU(param_norm);
    CHECK_CPU(grad_norm);
    CHECK_CPU(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        param_norm.type(),
        "compute_adaptive_lr_cpu",
        ([&] {
           ComputeAdaptiveLrAfterTypeCheck<scalar_t>(
               *param_norm.data<scalar_t>(),
               *grad_norm.data<scalar_t>(),
               weight_decay,
               eps,
               trust_coef,
               out.data<scalar_t>());
         }));
  }

  return out;
}

}
