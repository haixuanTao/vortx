//! Gaussian action sampler (GPU policy rollout). Added for zealot's
//! GPU-resident rollout (Stage 2): sample `a = mean + exp(log_std) * z` on-device
//! so the rollout loop never reads policy means back to the host.
//!
//! The noise `z` comes from a **counter-based RNG** keyed by
//! `(seed, env, dim, step)` — stateless, so the host can recompute the identical
//! value for any element without replaying a stream. This is what makes the
//! sample **pinnable**: with the same params the GPU and a host reference produce
//! bit-identical actions (modes 1 and 2 below use only integer + IEEE +-*/ ops;
//! mode 0's Box-Muller uses transcendentals that may differ in the last ULP).
//!
//! `pin_mode`:
//! - `0` PRODUCTION: full Gaussian noise (Box-Muller). For real rollouts.
//! - `1` MEAN: `z = 0`, so `action = mean` exactly. Bulletproof bit-exact compare.
//! - `2` UNIFORM: `z = u - 0.5` from the counter RNG (bounded, reproducible,
//!   bit-exact host↔device). A stochastic-but-comparable sample.

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
const TWO_PI: f32 = 6.283_185_5;
const HALF_LN_2PI: f32 = 0.918_938_5;

/// Scalar parameters for one sampler dispatch (uniform buffer; padded to 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct SampleParams {
    pub num_envs: u32,
    pub action_dim: u32,
    /// Global RNG seed (fixed per run → reproducible rollouts).
    pub seed: u32,
    /// Rollout step index (part of the RNG counter → independent noise per step).
    pub step: u32,
    /// 0 = Gaussian, 1 = mean (no noise), 2 = uniform bit-exact. See module docs.
    pub pin_mode: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Integer avalanche mix (Murmur3-style finalizer). Pure `u32` wrapping ops, so
/// it is identical on host and device.
#[inline]
pub fn mix(mut z: u32) -> u32 {
    z = (z ^ (z >> 16)).wrapping_mul(0x7feb_352d);
    z = (z ^ (z >> 15)).wrapping_mul(0x846c_a68b);
    z ^ (z >> 16)
}

/// Stateless counter-based RNG: hash `(seed, env, dim, step, draw)` to a `u32`.
#[inline]
pub fn counter_u32(seed: u32, env: u32, dim: u32, step: u32, draw: u32) -> u32 {
    let mut h = seed.wrapping_add(0x9e37_79b9);
    h = mix(h ^ env.wrapping_mul(0x85eb_ca6b));
    h = mix(h ^ dim.wrapping_mul(0xc2b2_ae35));
    h = mix(h ^ step.wrapping_mul(0x27d4_eb2f));
    h = mix(h ^ draw.wrapping_mul(0x1656_67b1));
    h
}

/// `u32` → `f32` in `[0, 1)` using the top 24 bits (exact; mantissa-aligned).
#[inline]
pub fn u01(h: u32) -> f32 {
    ((h >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// Noise draw for one (env, dim) element under the selected `pin_mode`.
/// Returns the standardized noise `z` (so `action = mean + exp(log_std) * z`).
#[inline]
pub fn noise_z(p: &SampleParams, env: u32, dim: u32) -> f32 {
    if p.pin_mode == 1 {
        // MEAN: deterministic, action == mean.
        0.0
    } else if p.pin_mode == 2 {
        // UNIFORM: bounded, bit-exact (only int->f32 and subtraction).
        u01(counter_u32(p.seed, env, dim, p.step, 0)) - 0.5
    } else {
        // GAUSSIAN: Box-Muller. Transcendentals may differ in the last ULP
        // between host std and device libdevice — not guaranteed bit-exact.
        let u0 = u01(counter_u32(p.seed, env, dim, p.step, 0));
        let u1 = u01(counter_u32(p.seed, env, dim, p.step, 1));
        // clamp u0 away from 0 to avoid ln(0).
        let u0 = if u0 < 1e-7 { 1e-7 } else { u0 };
        let r = (-2.0 * u0.ln()).sqrt();
        r * (TWO_PI * u1).cos()
    }
}

/// Sample an action vector per env: `a = mean + exp(log_std) * z`, and accumulate
/// the Gaussian log-prob of the drawn action per env. One thread per env; the
/// per-env loop over `action_dim` keeps the log-prob reduction thread-local
/// (no atomics, no barriers — embarrassingly parallel).
///
/// Layouts (all row-major, dense): `means`/`actions` are `[num_envs * action_dim]`
/// (env-major: element `env*action_dim + d`); `log_std` is `[action_dim]`
/// (state-independent, broadcast over envs); `logp` is `[num_envs]`.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gpu_sample_gaussian(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SampleParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] means: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] log_std: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] actions: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] logp: &mut [f32],
) {
    let adim = params.action_dim;
    for env in (invocation_id.x..params.num_envs).step_by(MAX_NUM_THREADS as usize) {
        let base = (env * adim) as usize;
        let mut lp = 0.0f32;
        for d in 0..adim {
            let i = base + d as usize;
            let m = means.read(i);
            let ls = log_std.read(d as usize);
            let std = ls.exp();
            let z = noise_z(params, env, d);
            *actions.at_mut(i) = m + std * z;
            // log N(a; m, std) = -0.5 z^2 - log_std - 0.5 ln(2pi), since (a-m)/std = z.
            lp += -0.5 * z * z - ls - HALF_LN_2PI;
        }
        *logp.at_mut(env as usize) = lp;
    }
}

/// Host CPU reference for `gpu_sample_gaussian` — mirrors the kernel exactly,
/// reusing the SAME `noise_z`/`exp`/log-prob ops, so for `pin_mode` 1 and 2 the
/// result is bit-identical to the GPU (the integer RNG and IEEE +-*/ are
/// deterministic across host and device). For `pin_mode` 0 the Box-Muller
/// transcendentals (`ln`/`cos`/`sqrt`) may differ in the last ULP between host
/// std and device libdevice. Fills `actions` `[num_envs*action_dim]` and `logp`
/// `[num_envs]`.
#[cfg(not(any(target_arch = "spirv", target_arch = "nvptx64")))]
pub fn cpu_sample(p: &SampleParams, means: &[f32], log_std: &[f32], actions: &mut [f32], logp: &mut [f32]) {
    let adim = p.action_dim as usize;
    for env in 0..p.num_envs {
        let base = env as usize * adim;
        let mut lp = 0.0f32;
        for d in 0..adim {
            let i = base + d;
            let m = means[i];
            let ls = log_std[d];
            let std = ls.exp();
            let z = noise_z(p, env, d as u32);
            actions[i] = m + std * z;
            lp += -0.5 * z * z - ls - HALF_LN_2PI;
        }
        logp[env as usize] = lp;
    }
}
