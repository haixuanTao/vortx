//! Linear algebra modules for shaders.

pub mod activation;
pub mod contiguous;
pub mod gemm;
pub mod inv;
// obs/reward are GPU-resident rollout kernels for the WebGPU path; on native
// CUDA the observation/reward assembly runs on the host, and their trig
// (atan2) has no cuda-oxide (nvptx) lowering — so exclude them from the device
// cubin build (they stay available for SPIR-V + host).
#[cfg(not(target_arch = "nvptx64"))]
pub mod obs;
pub mod op_assign;
pub mod optim;
pub mod ppo;
pub mod reduce;
pub mod repeat;
#[cfg(not(target_arch = "nvptx64"))]
pub mod reward;
pub mod sample;
pub mod shape;

pub use shape::Shape;
#[cfg(feature = "push_constants")]
pub use shape::{Shapes1, Shapes2, Shapes3};

// Re-export generated ShaderArgs structs (only available on host)
#[cfg(not(target_arch_is_gpu))]
pub use activation::{GpuElu, GpuEluBackward, GpuEluVec4, GpuTanh, GpuTanhBackward};
#[cfg(not(target_arch_is_gpu))]
pub use contiguous::{Contiguous, ContiguousWithOffset};
#[cfg(not(target_arch_is_gpu))]
pub use gemm::{GemmNaive, GemmTiled, GemmTiledVec4};
#[cfg(not(target_arch_is_gpu))]
pub use obs::GpuObs;
#[cfg(not(target_arch_is_gpu))]
pub use reward::GpuReward;
#[cfg(not(target_arch_is_gpu))]
pub use op_assign::{GpuAdd, GpuCopy, GpuCopyWithOffsets, GpuDiv, GpuMul, GpuSub};
#[cfg(not(target_arch_is_gpu))]
pub use optim::GpuAdam;
#[cfg(not(target_arch_is_gpu))]
pub use ppo::{GpuPpoActorGrad, GpuPpoValueGrad};
#[cfg(not(target_arch_is_gpu))]
pub use reduce::*;
#[cfg(not(target_arch_is_gpu))]
pub use repeat::Repeat;
#[cfg(not(target_arch_is_gpu))]
pub use sample::GpuSampleGaussian;
