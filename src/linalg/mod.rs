//! Fundamental linear-algebra matrix/vector operations.

mod activation;
mod contiguous;
mod gemm;
mod obs;
mod op_assign;
mod optim;
mod ppo;
mod reduce;
mod repeat;
mod reward;
mod sample;

pub use activation::Activation;
pub use contiguous::Contiguous;
pub use gemm::{Gemm, MatrixMode, N, T};
pub use obs::{JointObsCfg, Obs, ObsParams};
pub use op_assign::{BinOpOffsets, OpAssign, OpAssignVariant};
pub use optim::{Adam, AdamParams};
pub use ppo::{Ppo, PpoActorParams, PpoValueParams};
pub use reduce::{Reduce, ReduceVariant};
pub use repeat::Repeat;
pub use reward::{Reward, RewardJointCfg, RewardParams};
pub use sample::{Sample, SampleParams};

/// Returns the components of an nalgebra matrix (column-major) as a row-major buffer.
#[cfg(test)]
pub fn to_row_major(mat: &nalgebra::DMatrix<f32>) -> Vec<f32> {
    mat.transpose().as_slice().to_vec()
}
