//! Linear algebra modules for shaders.

pub mod activation;
pub mod contiguous;
pub mod gemm;
pub mod inv;
pub mod op_assign;
pub mod optim;
pub mod ppo;
pub mod reduce;
pub mod repeat;
pub mod sample;
pub mod shape;

pub use shape::Shape;
#[cfg(feature = "push_constants")]
pub use shape::{Shapes1, Shapes2, Shapes3};

// Re-export generated ShaderArgs structs (only available on host)
#[cfg(not(target_arch_is_gpu))]
pub use activation::{GpuTanh, GpuTanhBackward};
#[cfg(not(target_arch_is_gpu))]
pub use contiguous::{Contiguous, ContiguousWithOffset};
#[cfg(not(target_arch_is_gpu))]
pub use gemm::{GemmNaive, GemmTiled};
#[cfg(not(target_arch_is_gpu))]
pub use activation::{GpuElu, GpuEluBackward, GpuTanh, GpuTanhBackward};
#[cfg(not(target_arch_is_gpu))]
pub use gemm::{GemmNaive, GemmTiled, GemmTiledVec4};
pub use op_assign::{GpuAdd, GpuCopy, GpuCopyWithOffsets, GpuDiv, GpuMul, GpuSub};
#[cfg(not(target_arch_is_gpu))]
pub use reduce::*;
#[cfg(not(target_arch_is_gpu))]
pub use optim::GpuAdam;
#[cfg(not(target_arch_is_gpu))]
pub use ppo::{GpuPpoActorGrad, GpuPpoValueGrad};
#[cfg(not(target_arch_is_gpu))]
pub use activation::GpuEluVec4;
pub use reduce::{ReduceAdd, ReduceMax, ReduceMin, ReduceMul, ReduceSqNorm};
pub use repeat::Repeat;
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub use sample::GpuSampleGaussian;
