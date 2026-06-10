//! Diagonal-Gaussian action sampling + PD-target mapping (rollout, GPU-resident).
//!
//! Added for zealot's "whole-pipeline CUDA-graph" work: the rollout used to read
//! the actor `means` back to the CPU, draw `rng.gauss()` per element, map to PD
//! targets, then upload the targets again — three host round-trips that break
//! CUDA-graph capture. This kernel does all of it on-GPU from a counter-based RNG
//! (no per-step host noise upload), so the rollout inner loop has no host op.
//!
//! For env column `e` (one thread, looping over the small action dim `k`):
//!   noise   = N(0,1) from `hash(seed, e·A + k)`  (Box–Muller)
//!   action  = mean + exp(log_std)·noise          (matches `ActorCritic::sample`)
//!   target  = default_pos + action_scale·action  (matches `joint_targets`)
//! Every per-env tensor is row-major `[action_dim x num_envs]`; element `(k,e)`
//! lives at `k·num_envs + e`. `log_std` / `default_pos` / `action_scale` are
//! `[action_dim]`.

use crate::utils::limits::MAX_NUM_WORKGROUPS;
use glamx::UVec3;
use khal_std::{
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};
#[cfg(any(target_arch = "spirv", target_arch = "nvptx64"))]
use khal_std::num_traits::Float;

const WORKGROUP_SIZE: u32 = 256;
const MAX_NUM_THREADS: u32 = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

/// `2·π`, for the Box–Muller angle.
const TWO_PI: f32 = 6.283_185_5;
/// `1 / 2^32`, to map a `u32` hash into `[0, 1)`.
const INV_U32: f32 = 2.328_306_4e-10;

/// Scalar parameters for action sampling (uniform buffer; 16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct SampleParams {
    /// Action dimensionality (rows).
    pub action_dim: u32,
    /// Number of env columns `n`.
    pub num_envs: u32,
    /// Per-step RNG seed (advance once per rollout step for fresh noise).
    pub seed: u32,
    pub pad0: u32,
}

/// MurmurHash3 32-bit finalizer — a cheap, well-mixed integer hash. Deterministic
/// and identical on every backend (pure integer ops), so the CPU can reproduce
/// the exact draw for validation.
#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// One standard-normal draw keyed by `(seed, lane)` via Box–Muller over two
/// independent hashed uniforms. `u1` is nudged off zero so `ln(u1)` is finite.
#[inline]
fn gauss(seed: u32, lane: u32) -> f32 {
    let h1 = fmix32(seed ^ fmix32(lane.wrapping_mul(2)));
    let h2 = fmix32(seed ^ fmix32(lane.wrapping_mul(2).wrapping_add(1)));
    let mut u1 = h1 as f32 * INV_U32;
    if u1 < 1e-7 {
        u1 = 1e-7;
    }
    let u2 = h2 as f32 * INV_U32;
    (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos()
}

/// Sample actions and PD targets for every env, fully on-GPU.
///
/// Writes `action`, `target`, and the raw `noise` draws (all `[action_dim x
/// num_envs]`). `noise` is exposed so the host can recompute log-probs and so a
/// validator can reconstruct `action`/`target` bit-for-bit from the same draws.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_sample_targets(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SampleParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mean: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] log_std: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] default_pos: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] action_scale: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] action: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] target: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] noise: &mut [f32],
) {
    let a = params.action_dim as usize;
    let n = params.num_envs as usize;
    let seed = params.seed;
    for e in (invocation_id.x as usize..n).step_by(MAX_NUM_THREADS as usize) {
        for k in 0..a {
            let idx = k * n + e;
            // Lane keyed by (env, dim) so every element draws independently and
            // the same (e,k) is reproducible from `seed` alone.
            let z = gauss(seed, (e * a + k) as u32);
            let std = log_std.read(k).exp();
            let act = mean.read(idx) + std * z;
            *noise.at_mut(idx) = z;
            *action.at_mut(idx) = act;
            *target.at_mut(idx) = default_pos.read(k) + action_scale.read(k) * act;
        }
    }
}
