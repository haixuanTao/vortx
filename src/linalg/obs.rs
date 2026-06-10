//! Observation-assembly host dispatch. Added for zealot's GPU-resident rollout
//! (Stage 3a). Derives the per-env policy + critic observation from `body_poses`
//! on-GPU. Like the PPO/sample kernels it has no `Shape` uniform — dimensions and
//! topology ride in the params/cfg buffers — so no `TensorLayoutBuffers` is
//! needed. One GPU thread per env.

use crate::shaders::linalg::GpuObs;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

// Re-export the param/cfg structs from the shader crate.
pub use vortx_shaders::linalg::obs::{JointObsCfg, ObsParams};

/// GPU observation assembly (policy + privileged critic obs from poses).
#[derive(Shader)]
pub struct Obs {
    /// Derive `RobotState` from poses and write `obs`/`critic_obs`.
    pub obs: GpuObs,
}

impl Obs {
    /// Assemble observations for all envs. `poses`/`prev_poses` are raw pose
    /// buffers (8 f32/pose, `colliders_per_batch` poses per env). `joint_cfg` is
    /// `[J]`; `cmd` is `[3 x n]`; `last_action`/`prev_joint_pos` are `[J x n]`;
    /// `flags` is `[n]` u32 (bit0 has_prev_pose, bit1 has_prev_joint_pos). Writes
    /// `obs` `[obs_dim x n]`, `critic_obs` `[critic_obs_dim x n]`, `joint_pos_out`
    /// `[J x n]`. `params.num_envs` sets the thread count.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<ObsParams>,
        poses: impl AsTensorRef<f32>,
        prev_poses: impl AsTensorRef<f32>,
        joint_cfg: impl AsTensorRef<JointObsCfg>,
        cmd: impl AsTensorRef<f32>,
        last_action: impl AsTensorRef<f32>,
        prev_joint_pos: impl AsTensorRef<f32>,
        flags: impl AsTensorRef<u32>,
        mut obs: impl AsTensorMut<f32>,
        mut critic_obs: impl AsTensorMut<f32>,
        mut joint_pos_out: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let poses = poses.as_tensor_ref();
        let prev_poses = prev_poses.as_tensor_ref();
        let joint_cfg = joint_cfg.as_tensor_ref();
        let cmd = cmd.as_tensor_ref();
        let last_action = last_action.as_tensor_ref();
        let prev_joint_pos = prev_joint_pos.as_tensor_ref();
        let flags = flags.as_tensor_ref();
        let mut obs = obs.as_tensor_mut();
        let mut critic_obs = critic_obs.as_tensor_mut();
        let mut joint_pos_out = joint_pos_out.as_tensor_mut();

        // One thread per env column.
        let num_threads = flags.len() as u32;
        let mut buf_obs = obs.buffer_mut();
        let mut buf_critic = critic_obs.buffer_mut();
        let mut buf_jp = joint_pos_out.buffer_mut();

        self.obs.call(
            pass,
            num_threads,
            &params.buffer(),
            &poses.buffer(),
            &prev_poses.buffer(),
            &joint_cfg.buffer(),
            &cmd.buffer(),
            &last_action.buffer(),
            &prev_joint_pos.buffer(),
            &flags.buffer(),
            &mut buf_obs,
            &mut buf_critic,
            &mut buf_jp,
        )
    }
}
