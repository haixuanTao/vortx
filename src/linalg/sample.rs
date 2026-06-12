//! Gaussian action sampler host dispatch. Added for zealot's GPU-resident
//! rollout: sample `a = mean + exp(log_std) * z` on-device (counter-based RNG)
//! so the rollout loop never reads policy means back to the host. See the shader
//! crate module docs for the `pin_mode` semantics and the bit-exact guarantee.

use crate::shaders::linalg::GpuSampleGaussian;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};

// Re-export the params struct + the host reference from the shader crate.
pub use vortx_shaders::linalg::sample::{cpu_sample, SampleParams};

/// The Gaussian action sampler kernel.
#[derive(Shader)]
pub struct Sampler {
    /// Sample `a = mean + exp(log_std) * z`, one thread per env.
    pub sample: GpuSampleGaussian,
}

impl Sampler {
    /// Sample actions for `num_envs` envs. `params` is a scalar
    /// `Tensor<SampleParams>` (UNIFORM); `means`/`actions` are `[num_envs*action_dim]`
    /// (env-major); `log_std` is `[action_dim]`; `logp` is `[num_envs]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        _backend: &GpuBackend,
        pass: &mut GpuPass,
        num_envs: u32,
        params: impl AsTensorRef<SampleParams>,
        means: impl AsTensorRef<f32>,
        log_std: impl AsTensorRef<f32>,
        mut actions: impl AsTensorMut<f32>,
        mut logp: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let means = means.as_tensor_ref();
        let log_std = log_std.as_tensor_ref();
        let mut actions = actions.as_tensor_mut();
        let mut logp = logp.as_tensor_mut();

        let mut buf_actions = actions.buffer_mut();
        let mut buf_logp = logp.buffer_mut();

        self.sample.call(
            pass,
            num_envs,
            &params.buffer(),
            &means.buffer(),
            &log_std.buffer(),
            &mut buf_actions,
            &mut buf_logp,
        )
    }
}
