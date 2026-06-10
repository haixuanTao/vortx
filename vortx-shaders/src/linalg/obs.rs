//! GPU batched observation assembly for the velocity-tracking locomotion MDP.
//!
//! Stage 3a of the whole-pipeline-graph work: the rollout used to read the full
//! `body_poses` buffer back to the CPU and rebuild every env's policy/critic
//! observation in a rayon block (`read_state_from_poses` + `observe` /
//! `observe_critic`). That host round-trip breaks CUDA-graph capture. This kernel
//! derives the per-env `RobotState` from `body_poses` ON-GPU and writes the raw
//! (un-normalised) observation vectors, so the rollout's obs assembly has no host
//! op. Reward (the other half of the MDP) is a separate kernel (Stage 3b).
//!
//! Config-driven and physics-engine-agnostic: poses are read as raw `f32` with a
//! caller-supplied stride (nexus `Pose3` = 8 f32: rot xyzw at [0..4], translation
//! xyz at [4..7], pad at [7]); joint topology / defaults ride in `JointObsCfg`.
//! One GPU thread per env, looping over the (small) joint dim internally.
//!
//! Obs layout (matches `VelocityFlatTask::observe`):
//!   `[last_action(J), command(4), joint_pos_rel(J), joint_vel(J), proj_gravity(3)]`
//! Critic appends base linear & angular velocity in the body frame (`+6`). Every
//! output tensor is row-major `[dim x num_envs]`; element `(d,e)` at `d·n + e`.

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

/// Scalar configuration for observation assembly (uniform buffer; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct ObsParams {
    /// Number of envs `n` (thread count).
    pub num_envs: u32,
    /// Number of actuated joints `J`.
    pub num_joints: u32,
    /// Poses per env (`colliders_per_batch`) — stride between envs in `poses`.
    pub colliders_per_batch: u32,
    /// Link index of the torso/base within an env's pose block.
    pub torso_link: u32,
    /// Policy observation dimension (rows of `obs`).
    pub obs_dim: u32,
    /// Critic observation dimension (rows of `critic_obs`).
    pub critic_obs_dim: u32,
    /// Forward / lateral / up world-axis indices (FWD=0, LAT=1, UP=2).
    pub fwd: u32,
    pub lat: u32,
    pub up: u32,
    /// Control timestep (s) for the finite-difference velocities.
    pub control_dt: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
    pub pad4: u32,
    pub pad5: u32,
}

/// Per-joint topology + default for deriving joint angles (storage; 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct JointObsCfg {
    /// Parent link index (within an env's pose block).
    pub parent_link: u32,
    /// Child (actuated) link index.
    pub child_link: u32,
    /// Joint default position (rad), subtracted in `joint_pos_rel`.
    pub default_pos: f32,
    pub pad: f32,
    /// Joint rest quaternion `(x,y,z,w)` (parent→child at zero angle).
    pub rest_quat: [f32; 4],
}

// --- quaternion / vector helpers (match zealot-env/src/math.rs bit-for-bit) ---

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rotate `v` by unit quaternion `q=(x,y,z,w)` — `v + 2·q_xyz×(q_xyz×v + w·v)`.
#[inline]
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let tt = cross(u, t);
    [v[0] + 2.0 * tt[0], v[1] + 2.0 * tt[1], v[2] + 2.0 * tt[2]]
}

/// Rotate `v` by the inverse of `q` (world→body).
#[inline]
fn quat_rotate_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rotate([-q[0], -q[1], -q[2], q[3]], v)
}

/// Hamilton product `a·b` for `(x,y,z,w)` quaternions (matches glam `Quat * Quat`).
#[inline]
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline]
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Read the rotation quaternion `(x,y,z,w)` of pose `link` in env `e`.
#[inline]
fn pose_rot(poses: &[f32], e: usize, link: usize, cpb: usize) -> [f32; 4] {
    let base = (e * cpb + link) * 8;
    [
        poses.read(base),
        poses.read(base + 1),
        poses.read(base + 2),
        poses.read(base + 3),
    ]
}

/// Read the translation `(x,y,z)` of pose `link` in env `e`.
#[inline]
fn pose_trans(poses: &[f32], e: usize, link: usize, cpb: usize) -> [f32; 3] {
    let base = (e * cpb + link) * 8;
    [poses.read(base + 4), poses.read(base + 5), poses.read(base + 6)]
}

/// Assemble the policy + critic observation for every env, on-GPU.
///
/// `poses` / `prev_poses` are raw pose buffers (8 f32/pose). `flags.read(e)` bit0
/// = has_prev_pose (base velocities valid), bit1 = has_prev_joint_pos (joint
/// velocities valid). `cmd` is `[3 x n]` (vx, vy, yaw_rate). `last_action` /
/// `prev_joint_pos` are `[J x n]`. Writes `obs` `[obs_dim x n]`, `critic_obs`
/// `[critic_obs_dim x n]`, and `joint_pos_out` `[J x n]` (this step's joint
/// angles, to be committed as next step's `prev_joint_pos`).
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_obs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &ObsParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] poses: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] prev_poses: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] joint_cfg: &[JointObsCfg],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] cmd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] last_action: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] prev_joint_pos: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] flags: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] obs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] critic_obs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] joint_pos_out: &mut [f32],
) {
    let n = params.num_envs as usize;
    let j = params.num_joints as usize;
    let cpb = params.colliders_per_batch as usize;
    let torso = params.torso_link as usize;
    let od = params.obs_dim as usize;
    let cod = params.critic_obs_dim as usize;
    let (fwd, lat, up) = (params.fwd as usize, params.lat as usize, params.up as usize);
    let dt = params.control_dt;

    for e in (invocation_id.x as usize..n).step_by(MAX_NUM_THREADS as usize) {
        let fl = flags.read(e);
        let has_prev_pose = (fl & 1) != 0;
        let has_prev_jp = (fl & 2) != 0;

        // --- base orientation + finite-diff world velocities (torso link) ---
        let r = pose_rot(poses, e, torso, cpb);
        let t = pose_trans(poses, e, torso, cpb);
        let mut lin_w = [0.0f32; 3];
        let mut ang_w = [0.0f32; 3];
        if has_prev_pose {
            let pr = pose_rot(prev_poses, e, torso, cpb);
            let pt = pose_trans(prev_poses, e, torso, cpb);
            lin_w = [(t[0] - pt[0]) / dt, (t[1] - pt[1]) / dt, (t[2] - pt[2]) / dt];
            // ω ≈ 2·(Δq.xyz)/dt with hemisphere correction.
            let dq = quat_mul(r, quat_conj(pr));
            let s = if dq[3] >= 0.0 { 1.0 } else { -1.0 };
            ang_w = [2.0 * s * dq[0] / dt, 2.0 * s * dq[1] / dt, 2.0 * s * dq[2] / dt];
        }

        // --- projected gravity (world down rotated into the body frame) ---
        let mut world_down = [0.0f32; 3];
        world_down[up] = -1.0;
        let grav = quat_rotate_inv(r, world_down);

        // --- write the policy obs ([last_action, cmd, jp_rel, jvel, grav]) ---
        // last_action (J)
        for k in 0..j {
            *obs.at_mut(k * n + e) = last_action.read(k * n + e);
        }
        // command (4): [vx, vy, yaw_rate, 0]
        *obs.at_mut((j) * n + e) = cmd.read(e);
        *obs.at_mut((j + 1) * n + e) = cmd.read(n + e);
        *obs.at_mut((j + 2) * n + e) = cmd.read(2 * n + e);
        *obs.at_mut((j + 3) * n + e) = 0.0;
        // joint_pos_rel (J) + joint_vel (J), deriving the joint angle per joint
        let base_jp = j + 4;
        let base_jv = j + 4 + j;
        for k in 0..j {
            let cfg = joint_cfg.read(k);
            let qp = pose_rot(poses, e, cfg.parent_link as usize, cpb);
            let qc = pose_rot(poses, e, cfg.child_link as usize, cpb);
            // rel = rest⁻¹ · qp⁻¹ · qc ; θ = 2·atan2(rel.z, rel.w)
            let rel = quat_mul(quat_mul(quat_conj(cfg.rest_quat), quat_conj(qp)), qc);
            let theta = 2.0 * rel[2].atan2(rel[3]);
            *joint_pos_out.at_mut(k * n + e) = theta;
            *obs.at_mut((base_jp + k) * n + e) = theta - cfg.default_pos;
            let jv = if has_prev_jp {
                (theta - prev_joint_pos.read(k * n + e)) / dt
            } else {
                0.0
            };
            *obs.at_mut((base_jv + k) * n + e) = jv;
        }
        // projected gravity (3)
        let base_g = j + 4 + j + j;
        *obs.at_mut(base_g * n + e) = grav[0];
        *obs.at_mut((base_g + 1) * n + e) = grav[1];
        *obs.at_mut((base_g + 2) * n + e) = grav[2];

        // --- critic obs = policy obs (copied) + base lin/ang vel in body frame ---
        for d in 0..od {
            *critic_obs.at_mut(d * n + e) = obs.read(d * n + e);
        }
        let v_body = quat_rotate_inv(r, lin_w);
        let w_body = quat_rotate_inv(r, ang_w);
        *critic_obs.at_mut(od * n + e) = v_body[0];
        *critic_obs.at_mut((od + 1) * n + e) = v_body[1];
        *critic_obs.at_mut((od + 2) * n + e) = v_body[2];
        *critic_obs.at_mut((od + 3) * n + e) = w_body[0];
        *critic_obs.at_mut((od + 4) * n + e) = w_body[1];
        *critic_obs.at_mut((od + 5) * n + e) = w_body[2];
        // (cod is od+6; silence unused on the spirv path)
        let _ = (cod, lat, fwd);
    }
}
