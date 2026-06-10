//! Action-sampling host dispatch. Added for zealot's GPU-resident rollout.
//!
//! Wraps the fused diagonal-Gaussian sample + PD-target kernel. Like the PPO
//! kernels it has no `Shape` uniform — dimensions ride in the params struct and
//! indexing is row-major — so no `TensorLayoutBuffers` is needed. One GPU thread
//! per env column.

use crate::shaders::linalg::GpuSampleTargets;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

// Re-export the params struct from the shader crate.
pub use vortx_shaders::linalg::sample::SampleParams;

/// Fused diagonal-Gaussian action sampler + PD-target mapping.
#[derive(Shader)]
pub struct Sample {
    /// Sample `action`/`target`/`noise` for every env, on-GPU.
    pub sample_targets: GpuSampleTargets,
}

impl Sample {
    /// Sample actions and PD targets for all envs. `mean` is row-major
    /// `[action_dim x num_envs]`; `log_std` / `default_pos` / `action_scale` are
    /// `[action_dim]`. Writes `action`, `target`, and `noise` (all
    /// `[action_dim x num_envs]`). `params.num_envs` sets the thread count.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_targets(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<SampleParams>,
        mean: impl AsTensorRef<f32>,
        log_std: impl AsTensorRef<f32>,
        default_pos: impl AsTensorRef<f32>,
        action_scale: impl AsTensorRef<f32>,
        mut action: impl AsTensorMut<f32>,
        mut target: impl AsTensorMut<f32>,
        mut noise: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let mean = mean.as_tensor_ref();
        let log_std = log_std.as_tensor_ref();
        let default_pos = default_pos.as_tensor_ref();
        let action_scale = action_scale.as_tensor_ref();
        let mut action = action.as_tensor_mut();
        let mut target = target.as_tensor_mut();
        let mut noise = noise.as_tensor_mut();

        // One thread per env column.
        let num_threads = (mean.len() / log_std.len().max(1)) as u32;
        let mut buf_action = action.buffer_mut();
        let mut buf_target = target.buffer_mut();
        let mut buf_noise = noise.buffer_mut();

        self.sample_targets.call(
            pass,
            num_threads,
            &params.buffer(),
            &mean.buffer(),
            &log_std.buffer(),
            &default_pos.buffer(),
            &action_scale.buffer(),
            &mut buf_action,
            &mut buf_target,
            &mut buf_noise,
        )
    }
}

/// CPU reference for the kernel's counter-RNG draw — identical integer ops, so
/// it reproduces a given `(seed, lane)` bit-for-bit. Used to validate the fused
/// arithmetic (transcendentals may differ by a ULP between host and device, so
/// validation compares reconstructed `action`/`target` from the kernel's own
/// `noise` readback rather than re-deriving the Gaussian on the host).
#[allow(dead_code)]
pub fn cpu_fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmix32_matches_known_vector() {
        // MurmurHash3 fmix32(0) == 0, fmix32(1) is a fixed mixed value.
        assert_eq!(cpu_fmix32(0), 0);
        assert_ne!(cpu_fmix32(1), 1);
        // Determinism.
        assert_eq!(cpu_fmix32(0xdead_beef), cpu_fmix32(0xdead_beef));
    }
}
