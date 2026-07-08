//! GPU batched reward for the velocity-tracking locomotion MDP (Stage 3b).
//!
//! Companion to `gpu_obs`: derives the per-env `RobotState` + per-foot contact
//! state from `body_poses` ON-GPU and evaluates the 20-term weighted reward plus
//! the fall-termination penalty, so the rollout's reward computation has no host
//! op. Exact port of `VelocityFlatTask::reward` + `compute_feet_from_poses` +
//! `fell_over`. One GPU thread per env (loops over the small joint/foot dims).
//!
//! Self-contained (recomputes base velocity / gravity / joint angles rather than
//! reading `gpu_obs`'s outputs) so it can be validated standalone against the CPU
//! reward. Poses are raw f32, `Pose3` stride 8 (rot xyzw [0..4], trans [4..7]).
//! Writes `reward` `[n]` (incl. termination), `fell` `[n]` u32, and `new_air`
//! `[num_feet x n]` (committed as next step's air-time).

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
const MAX_FEET: usize = 2;

/// Scalar config + all reward weights/stds/thresholds (uniform buffer).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardParams {
    pub num_envs: u32,
    pub num_joints: u32,
    pub num_feet: u32,
    pub colliders_per_batch: u32,
    pub torso_link: u32,
    pub fwd: u32,
    pub lat: u32,
    pub up: u32,
    pub foot_link0: u32,
    pub foot_link1: u32,
    pub pad_u0: u32,
    pub pad_u1: u32,

    pub control_dt: f32,
    pub w_track_lin: f32,
    pub w_track_ang: f32,
    pub w_upright: f32,
    pub w_base_height: f32,
    pub base_height_target: f32,
    pub w_pose: f32,
    pub w_bilateral: f32,
    pub w_action_rate: f32,
    pub w_action_rate_hip: f32,
    pub w_body_ang_vel: f32,
    pub w_lin_vel_z: f32,
    pub w_dof_pos_limits: f32,
    pub w_dof_vel: f32,
    pub w_termination: f32,
    pub w_air_time: f32,
    pub w_flight: f32,
    pub w_single_support: f32,
    pub w_foot_slip: f32,
    pub w_foot_clearance: f32,
    pub foot_clearance_target: f32,
    pub w_foot_orientation: f32,
    pub w_feet_yaw_mean: f32,
    pub w_feet_distance: f32,
    pub feet_distance_ref: f32,

    pub std_lin_vel: f32,
    pub std_ang_vel: f32,
    pub std_upright: f32,
    pub std_base_height: f32,
    pub std_pose: f32,

    pub contact_z: f32,
    pub min_base_height: f32,
    pub tilt_cos: f32,
    pub standing_speed: f32,
    pub air_cap: f32,
    pub limit_scale: f32,
    pub pad_f0: f32,
    pub pad_f1: f32,
}

/// Per-joint reward config (storage; 64 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(
    not(any(target_arch = "spirv", target_arch = "nvptx64")),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct RewardJointCfg {
    pub parent_link: u32,
    pub child_link: u32,
    pub default_pos: f32,
    pub pos_lo: f32,
    pub pos_hi: f32,
    /// Index of the mirror partner joint (for bilateral symmetry).
    pub sym_partner: u32,
    /// Mirror sign (+1 sagittal, -1 lateral).
    pub sym_sign: f32,
    /// 1 if this (left) joint contributes a symmetry term.
    pub sym_active: u32,
    /// 1 if this joint is a hip yaw/roll DOF (for `action_rate_hipz_hipx`).
    pub is_hip: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    /// Joint rest quaternion `(x,y,z,w)`.
    pub rest_quat: [f32; 4],
}

// --- quaternion / vector helpers (match zealot-env/math.rs + glam) ---
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let t = cross(u, v);
    let t = [t[0] + w * v[0], t[1] + w * v[1], t[2] + w * v[2]];
    let tt = cross(u, t);
    [v[0] + 2.0 * tt[0], v[1] + 2.0 * tt[1], v[2] + 2.0 * tt[2]]
}
#[inline]
fn quat_rotate_inv(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    quat_rotate([-q[0], -q[1], -q[2], q[3]], v)
}
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
#[inline]
fn pose_rot(poses: &[f32], e: usize, link: usize, cpb: usize) -> [f32; 4] {
    let b = (e * cpb + link) * 8;
    [poses.read(b), poses.read(b + 1), poses.read(b + 2), poses.read(b + 3)]
}
#[inline]
fn pose_trans(poses: &[f32], e: usize, link: usize, cpb: usize) -> [f32; 3] {
    let b = (e * cpb + link) * 8;
    [poses.read(b + 4), poses.read(b + 5), poses.read(b + 6)]
}

/// Joint angle `θ = 2·atan2(rel.z, rel.w)`, `rel = rest⁻¹·qp⁻¹·qc`, from `poses`.
#[inline]
fn joint_angle(poses: &[f32], e: usize, cpb: usize, cfg: RewardJointCfg) -> f32 {
    let qp = pose_rot(poses, e, cfg.parent_link as usize, cpb);
    let qc = pose_rot(poses, e, cfg.child_link as usize, cpb);
    let rel = quat_mul(quat_mul(quat_conj(cfg.rest_quat), quat_conj(qp)), qc);
    2.0 * rel[2].atan2(rel[3])
}

/// World up basis vector (e.g. `[0,0,1]` for up=2) — for `upright_cos`.
#[inline]
fn up_basis(up: usize) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    v[up] = 1.0;
    v
}

/// Evaluate the per-env step reward (incl. termination) for every env, on-GPU.
#[spirv_bindgen]
#[spirv(compute(threads(256, 1, 1)))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_reward(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] p: &RewardParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] poses: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] prev_poses: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] joint_cfg: &[RewardJointCfg],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] cmd: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] action2: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] air_time_in: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sole_local: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] flags: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] reward: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] fell: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] new_air: &mut [f32],
) {
    let n = p.num_envs as usize;
    let j = p.num_joints as usize;
    let nf = p.num_feet as usize;
    let cpb = p.colliders_per_batch as usize;
    let torso = p.torso_link as usize;
    let (fwd, lat, up) = (p.fwd as usize, p.lat as usize, p.up as usize);
    let dt = p.control_dt;

    for e in (invocation_id.x as usize..n).step_by(MAX_NUM_THREADS as usize) {
        let fl = flags.read(e);
        let has_prev_pose = (fl & 1) != 0;
        let has_prev_jp = (fl & 2) != 0;

        // --- base state ---
        let r = pose_rot(poses, e, torso, cpb);
        let t = pose_trans(poses, e, torso, cpb);
        let mut lin_w = [0.0f32; 3];
        let mut ang_w = [0.0f32; 3];
        if has_prev_pose {
            let pr = pose_rot(prev_poses, e, torso, cpb);
            let pt = pose_trans(prev_poses, e, torso, cpb);
            lin_w = [(t[0] - pt[0]) / dt, (t[1] - pt[1]) / dt, (t[2] - pt[2]) / dt];
            let dq = quat_mul(r, quat_conj(pr));
            let s = if dq[3] >= 0.0 { 1.0 } else { -1.0 };
            ang_w = [2.0 * s * dq[0] / dt, 2.0 * s * dq[1] / dt, 2.0 * s * dq[2] / dt];
        }
        let v = quat_rotate_inv(r, lin_w);
        let w = quat_rotate_inv(r, ang_w);
        let mut world_down = [0.0f32; 3];
        world_down[up] = -1.0;
        let grav = quat_rotate_inv(r, world_down);
        let height = t[up];

        // command
        let cvx = cmd.read(e);
        let cvy = cmd.read(n + e);
        let cyaw = cmd.read(2 * n + e);
        let speed = (cvx * cvx + cvy * cvy + cyaw * cyaw).sqrt();
        let standing = speed < p.standing_speed;
        let moving = !standing;

        // --- tracking / upright / height ---
        let lin_err = (cvx - v[fwd]) * (cvx - v[fwd]) + (cvy - v[lat]) * (cvy - v[lat]);
        let track_lin = p.w_track_lin * (-lin_err / (p.std_lin_vel * p.std_lin_vel)).exp() * dt;
        let ang_err = (cyaw - w[up]) * (cyaw - w[up]);
        let track_ang = p.w_track_ang * (-ang_err / (p.std_ang_vel * p.std_ang_vel)).exp() * dt;
        let tilt_err = grav[fwd] * grav[fwd] + grav[lat] * grav[lat];
        let upright = p.w_upright * (-tilt_err / (p.std_upright * p.std_upright)).exp() * dt;
        let h_err = (height - p.base_height_target) * (height - p.base_height_target);
        let base_h = p.w_base_height * (-h_err / (p.std_base_height * p.std_base_height)).exp() * dt;

        // --- joint-dependent sums (angle, vel, pose, symmetry, limits, action) ---
        let mut pose_err = 0.0f32;
        let mut sym_err = 0.0f32;
        let mut lim_pen = 0.0f32;
        let mut jv2 = 0.0f32;
        let mut da2 = 0.0f32;
        let mut da2_hip = 0.0f32;
        for k in 0..j {
            let cfg = joint_cfg.read(k);
            let q = joint_angle(poses, e, cpb, cfg);
            let d = q - cfg.default_pos;
            pose_err += d * d;
            if cfg.sym_active != 0 {
                let qr = joint_angle(poses, e, cpb, joint_cfg.read(cfg.sym_partner as usize));
                let de = q - cfg.sym_sign * qr;
                sym_err += de * de;
            }
            let hi = cfg.pos_hi * p.limit_scale;
            let lo = cfg.pos_lo * p.limit_scale;
            lim_pen += (q - hi).max(0.0) + (lo - q).max(0.0);
            let jv = if has_prev_jp {
                // prev joint angle isn't stored here; reward uses the SAME
                // finite-diff as obs, but we only have prev_poses → recompute the
                // previous angle from prev_poses.
                let qpp = pose_rot(prev_poses, e, cfg.parent_link as usize, cpb);
                let qcp = pose_rot(prev_poses, e, cfg.child_link as usize, cpb);
                let relp = quat_mul(quat_mul(quat_conj(cfg.rest_quat), quat_conj(qpp)), qcp);
                let qprev = 2.0 * relp[2].atan2(relp[3]);
                (q - qprev) / dt
            } else {
                0.0
            };
            jv2 += jv * jv;
            let la = action2.read(k * n + e);
            let pa = action2.read((j + k) * n + e);
            let dadiff = la - pa;
            da2 += dadiff * dadiff;
            if cfg.is_hip != 0 {
                da2_hip += dadiff * dadiff;
            }
        }
        let pose = if standing {
            p.w_pose * (-pose_err / (p.std_pose * p.std_pose)).exp() * dt
        } else {
            0.0
        };
        let bilateral = p.w_bilateral * (-sym_err).exp() * dt;
        let action_rate = p.w_action_rate * da2 * dt;
        let action_rate_hip = p.w_action_rate_hip * da2_hip * dt;
        let body_ang_vel = p.w_body_ang_vel * (w[fwd] * w[fwd] + w[lat] * w[lat]) * dt;
        let lin_vel_z = p.w_lin_vel_z * v[up] * v[up] * dt;
        let dof_pos_limits = p.w_dof_pos_limits * lim_pen * dt;
        let dof_vel = p.w_dof_vel * jv2 * dt;

        // --- feet ---
        let base_rot_inv = quat_conj(r);
        let mut air_sum = 0.0f32;
        let mut all_air = true;
        let mut contacts = 0u32;
        let mut slip = 0.0f32;
        let mut clr = 0.0f32;
        let mut tilt_sq = 0.0f32;
        let mut yaw_sq = 0.0f32;
        // foot positions cached for the lateral-distance term (up to MAX_FEET).
        let mut fx = [0.0f32; MAX_FEET];
        let mut fy = [0.0f32; MAX_FEET];
        for i in 0..nf {
            let link = if i == 0 { p.foot_link0 } else { p.foot_link1 } as usize;
            let fpos = pose_trans(poses, e, link, cpb);
            let frot = pose_rot(poses, e, link, cpb);
            let planar_speed = if has_prev_pose {
                let pp = pose_trans(prev_poses, e, link, cpb);
                let dx = (fpos[0] - pp[0]) / dt;
                let dy = (fpos[1] - pp[1]) / dt;
                (dx * dx + dy * dy).sqrt()
            } else {
                0.0
            };
            let sole = [
                sole_local.read((i * 3) * n + e),
                sole_local.read((i * 3 + 1) * n + e),
                sole_local.read((i * 3 + 2) * n + e),
            ];
            let world_normal = quat_rotate(frot, sole);
            let tilt = world_normal[2].abs().min(1.0).max(0.0).acos();
            let fx_base = quat_rotate(quat_mul(base_rot_inv, frot), [1.0, 0.0, 0.0]);
            let yaw_rel = fx_base[1].atan2(fx_base[0]);
            let contact = fpos[2] < p.contact_z;
            let prev_air = air_time_in.read(i * n + e);
            let first_contact = contact && prev_air > 0.0;
            let na = if contact { 0.0 } else { prev_air + dt };
            *new_air.at_mut(i * n + e) = na;
            let air_time = if contact { prev_air } else { na };

            if contact {
                contacts += 1;
                slip += planar_speed * planar_speed;
                tilt_sq += tilt * tilt;
                all_air = false;
            } else {
                clr += (fpos[2] - p.foot_clearance_target) * (fpos[2] - p.foot_clearance_target)
                    * planar_speed;
            }
            if first_contact {
                air_sum += if air_time < p.air_cap { air_time } else { p.air_cap };
            }
            yaw_sq += yaw_rel * yaw_rel;
            if i < MAX_FEET {
                fx[i] = fpos[0];
                fy[i] = fpos[1];
            }
        }
        let air_time = if moving { p.w_air_time * air_sum * dt } else { 0.0 };
        let flight = if all_air { p.w_flight * dt } else { 0.0 };
        let single_support = if moving && contacts == 1 { p.w_single_support * dt } else { 0.0 };
        let foot_slip = p.w_foot_slip * slip * dt;
        let foot_clearance = p.w_foot_clearance * clr * dt;
        let foot_orientation = p.w_foot_orientation * tilt_sq * dt;
        let feet_yaw_mean = p.w_feet_yaw_mean * yaw_sq * dt;
        let feet_distance = if nf == 2 {
            let dx = fx[0] - fx[1];
            let dy = fy[0] - fy[1];
            let base_yaw = (2.0 * (r[3] * r[2] + r[0] * r[1]))
                .atan2(1.0 - 2.0 * (r[1] * r[1] + r[2] * r[2]));
            let cy = base_yaw.cos();
            let sy = base_yaw.sin();
            let lateral = -sy * dx + cy * dy;
            let err = lateral.abs() - p.feet_distance_ref;
            p.w_feet_distance * err.abs() * dt
        } else {
            0.0
        };

        let mut total = track_lin
            + track_ang
            + upright
            + base_h
            + pose
            + bilateral
            + action_rate
            + action_rate_hip
            + body_ang_vel
            + lin_vel_z
            + dof_pos_limits
            + dof_vel
            + air_time
            + flight
            + single_support
            + foot_slip
            + foot_clearance
            + foot_orientation
            + feet_yaw_mean
            + feet_distance;

        // --- termination ---
        let up_axis = quat_rotate(r, up_basis(up));
        let upright_cos = up_axis[up];
        // `height - height == 0` is true iff height is finite (inf/NaN → NaN ≠ 0).
        // Avoids emitting an infinite float constant, which naga rejects in SPIR-V.
        let non_finite = (height - height) != 0.0;
        let fell_over =
            non_finite || height < p.min_base_height || upright_cos < p.tilt_cos;
        if fell_over {
            total += p.w_termination;
            *fell.at_mut(e) = 1;
        } else {
            *fell.at_mut(e) = 0;
        }
        *reward.at_mut(e) = total;
    }
}
