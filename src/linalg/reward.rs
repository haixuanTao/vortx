//! Reward host dispatch. Added for zealot's GPU-resident rollout (Stage 3b).
//! Evaluates the velocity-tracking locomotion reward (20 weighted terms +
//! termination) from `body_poses` on-GPU. No `Shape` uniform — config rides in
//! the params/cfg buffers. One GPU thread per env.

use crate::shaders::linalg::GpuReward;
use crate::tensor::{AsTensorMut, AsTensorRef};
use khal::Shader;
use khal::backend::{GpuBackendError, GpuPass};

// Re-export the param/cfg structs from the shader crate.
pub use vortx_shaders::linalg::reward::{RewardJointCfg, RewardParams};

/// GPU step-reward evaluation for the velocity-tracking MDP.
#[derive(Shader)]
pub struct Reward {
    /// 20-term weighted reward + fall termination, per env.
    pub reward: GpuReward,
}

impl Reward {
    /// Evaluate the per-env reward. `poses`/`prev_poses` are raw pose buffers
    /// (8 f32/pose). `joint_cfg` is `[J]`; `cmd` is `[3 x n]`; `action2` is
    /// `[2J x n]` (last_action rows `0..J`, prev_action rows `J..2J`);
    /// `air_time_in`/`new_air` are `[num_feet x n]`; `sole_local` is
    /// `[num_feet*3 x n]` (foot-local sole normal per env); `flags` is `[n]` u32.
    /// Writes `reward` `[n]` (incl. termination penalty), `fell` `[n]` u32.
    /// `params.num_envs` sets the thread count.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        pass: &mut GpuPass,
        params: impl AsTensorRef<RewardParams>,
        poses: impl AsTensorRef<f32>,
        prev_poses: impl AsTensorRef<f32>,
        joint_cfg: impl AsTensorRef<RewardJointCfg>,
        cmd: impl AsTensorRef<f32>,
        action2: impl AsTensorRef<f32>,
        air_time_in: impl AsTensorRef<f32>,
        sole_local: impl AsTensorRef<f32>,
        flags: impl AsTensorRef<u32>,
        mut reward: impl AsTensorMut<f32>,
        mut fell: impl AsTensorMut<u32>,
        mut new_air: impl AsTensorMut<f32>,
    ) -> Result<(), GpuBackendError> {
        let params = params.as_tensor_ref();
        let poses = poses.as_tensor_ref();
        let prev_poses = prev_poses.as_tensor_ref();
        let joint_cfg = joint_cfg.as_tensor_ref();
        let cmd = cmd.as_tensor_ref();
        let action2 = action2.as_tensor_ref();
        let air_time_in = air_time_in.as_tensor_ref();
        let sole_local = sole_local.as_tensor_ref();
        let flags = flags.as_tensor_ref();
        let mut reward = reward.as_tensor_mut();
        let mut fell = fell.as_tensor_mut();
        let mut new_air = new_air.as_tensor_mut();

        let num_threads = flags.len() as u32; // one thread per env
        let mut buf_reward = reward.buffer_mut();
        let mut buf_fell = fell.buffer_mut();
        let mut buf_new_air = new_air.buffer_mut();

        self.reward.call(
            pass,
            num_threads,
            &params.buffer(),
            &poses.buffer(),
            &prev_poses.buffer(),
            &joint_cfg.buffer(),
            &cmd.buffer(),
            &action2.buffer(),
            &air_time_in.buffer(),
            &sole_local.buffer(),
            &flags.buffer(),
            &mut buf_reward,
            &mut buf_fell,
            &mut buf_new_air,
        )
    }
}
