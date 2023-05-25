import os
from typing import Tuple
import torch
import numpy as np
from torch import Tensor
from gym.spaces import Box
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, get_euler_xyz
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import quat_diff_rad
from itertools import combinations

BLUE_TEAM = 0
YELLOW_TEAM = 1
RED_ROBOT = 0
GREEN_ROBOT = 1
PINK_ROBOT = 2
ORANGE_COLOR = gymapi.Vec3(1.0, 0.4, 0.0)
BLUE_COLOR = gymapi.Vec3(0.0, 0.0, 1.0)
YELLOW_COLOR = gymapi.Vec3(1.0, 1.0, 0.0)
RED_COLOR = gymapi.Vec3(1.0, 0.0, 0.0)
GREEN_COLOR = gymapi.Vec3(0.0, 1.0, 0.0)
PINK_COLOR = gymapi.Vec3(1.0, 0.0, 1.0)
TEAM_COLORS = [BLUE_COLOR, YELLOW_COLOR]
ID_COLORS = [RED_COLOR, GREEN_COLOR, PINK_COLOR]
NUM_TEAMS = 1
NUM_ROBOTS = 1


class VSSGoTo(VecTask):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 20}
    #####################################################################
    ###==============================init=============================###
    #####################################################################
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.max_episode_length = cfg['env']['maxEpisodeLength']
        self.w_terminal = cfg['env']['rew_weights']['terminal']
        self.w_move = cfg['env']['rew_weights']['move']
        self.dist_threshold = cfg['env']['thresholds']['dist']
        self.angle_threshold = cfg['env']['thresholds']['angle']
        self.speed_threshold = cfg['env']['thresholds']['speed']
        self.robot_max_wheel_rad_s = 42.0
        self.min_robot_placement_dist = 0.07
        self.cfg = cfg
        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.7, 1.09, 4.6)
            cam_target = gymapi.Vec3(1.7, 1.1, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self._acquire_tensors()
        self._refresh_tensors()
        self.reset_dones()
        self.compute_observations()

    def _acquire_tensors(self):
        """Acquire and wrap tensors. Create views."""
        n_field_actors = 8  # 2 side walls, 4 end walls, 2 goal walls
        total_robots = NUM_ROBOTS * NUM_TEAMS
        num_actors = total_robots * 2 + n_field_actors
        self.s_robots = slice(0, total_robots)
        self.s_tgts = slice(total_robots, total_robots+total_robots)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state).view(
            self.num_envs, num_actors, 13
        )

        self.root_pos = self.root_state[..., 0:2]
        self.robots_pos = self.root_pos[:, self.s_robots, :]
        self.tgts_pos = self.root_pos[:, self.s_tgts, :]

        self.root_quats = self.root_state[..., 3:7]
        self.robots_quats = self.root_quats[:, self.s_robots, :]
        self.tgts_quats = self.root_quats[:, self.s_tgts, :]
        self.root_vel = self.root_state[..., 7:9]
        self.robots_vel = self.root_vel[:, self.s_robots, :]
        self.root_ang_vel = self.root_state[..., 12:13]
        self.robots_ang_vel = self.root_ang_vel[:, self.s_robots]
        for i in range(50):
            self.gym.simulate(self.sim)
        self._refresh_tensors()
        self.env_reset_root_state = self.root_state.mean(0).clone()

        self.dof_velocity_buf = torch.zeros(
            (self.num_envs, total_robots, 2),
            device=self.device,
            requires_grad=False,
        )
        self.field_scale = torch.tensor(
            [self.field_width, self.field_height],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) - (self.min_robot_placement_dist * 2)
        self.field_size = torch.tensor(
            [self.field_width + 2*self.goal_width, self.field_height],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.grad_offset = torch.tensor(
            [self.goal_width, 0.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # Add goal depth to grad calculation to decrease goal center weight
        self.yellow_goal = torch.tensor(
            [self.field_width / 2, 0.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.z_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float, device=self.device, requires_grad=False
        )
        entities_ids = list(range(2))  # 2 = Robot + Target
        self.entities_pairs = torch.tensor(
            list(combinations(entities_ids, 2)), device=self.device, requires_grad=False
        )

    #####################################################################
    ###==============================step=============================###
    #####################################################################
    def pre_physics_step(self, _actions):
        # reset progress_buf for envs reseted on previous step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.progress_buf[env_ids] = 0
        self.dof_velocity_buf[:] = _actions.to(self.device).unsqueeze(1)

        act = self.dof_velocity_buf * self.robot_max_wheel_rad_s
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(act))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_rewards_and_dones()

        # Save observations previously to resets
        self.compute_observations()
        self.extras["terminal_observation"] = self.obs_buf.clone().to(self.rl_device)
        self.extras["progress_buffer"] = self.progress_buf.clone().to(self.rl_device).float()

        self.reset_dones()
        self.compute_observations()

    def compute_observations(self):
        self.obs_buf[:] = compute_goto_obs(
            self.tgts_pos,
            self.robots_pos,
            self.robots_vel,
            self.robots_quats,
            self.robots_ang_vel,
            self.dof_velocity_buf
        )

    def compute_rewards_and_dones(self):
        self.rew_buf *= 0
        
        prev_tgt_dist = calculate_distance(self.tgts_pos, self.robots_pos)
        self._refresh_tensors()

        tgt_dist = calculate_distance(self.tgts_pos, self.robots_pos)
        angle_diff = calculate_angle_diff(self.tgts_quats, self.robots_quats)
        robot_speed = self.robots_vel.norm(dim=-1).squeeze()

        self.extras["log"] = {
            "angle_diff": angle_diff.clone().to(self.rl_device),
            "robot_speed": robot_speed.clone().to(self.rl_device),
        }

        terminal = check_terminal_state(
            self.reset_buf,
            tgt_dist,
            angle_diff,
            robot_speed,
            self.dist_threshold,
            self.angle_threshold,
            self.speed_threshold,
        )

        self.reset_buf[:] = compute_goto_dones(
            terminal,
            self.progress_buf,
            self.max_episode_length,
        )

        # Move Rewards
        if self.w_move > 0:
            move_rew = (prev_tgt_dist - tgt_dist) * self.w_move
            self.rew_buf[:] += move_rew.squeeze()

        if self.w_terminal > 0:
            self.rew_buf[:] += terminal.squeeze() * self.w_terminal
        

    def reset_dones(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            # Reset env state
            self.root_state[env_ids] = self.env_reset_root_state
            close_ids = torch.arange(len(env_ids), device=self.device)
            rand_pos = torch.zeros(
                (len(env_ids), 2, 2),  # 2 = 1 Robots + Target, 2 = X,Y
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            while len(close_ids):
                # randomize positions
                rand_pos[close_ids] = (
                    torch.rand(
                        (len(close_ids), 2, 2),  # 2 = 1 Robots + Target, 2 = X,Y
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    - 0.5
                ) * self.field_scale
                # Check for colliding robots
                dists = torch.linalg.norm(
                    rand_pos[:, self.entities_pairs[:, 0], :]
                    - rand_pos[:, self.entities_pairs[:, 1], :],
                    dim=2,
                )
                too_close = torch.any(dists < self.min_robot_placement_dist, dim=1)
                close_ids = too_close.nonzero(as_tuple=False).squeeze(-1)

            self.robots_pos[env_ids] = rand_pos[:, self.s_robots]
            self.tgts_pos[env_ids] = (rand_pos[:, self.s_tgts] / self.field_scale) * self.field_size

            # randomize rotations
            rand_angles = torch_rand_float(
                -np.pi,
                np.pi,
                (len(env_ids), NUM_TEAMS * NUM_ROBOTS),
                device=self.device,
            )
            self.robots_quats[env_ids] = quat_from_angle_axis(
                rand_angles, self.z_axis
            )
            # randomize rotations
            rand_angles = torch_rand_float(
                -np.pi,
                np.pi,
                (len(env_ids), NUM_TEAMS * NUM_ROBOTS),
                device=self.device,
            )
            self.tgts_quats[env_ids] = quat_from_angle_axis(
                rand_angles, self.z_axis
            )

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_state)
            )

            self.dof_velocity_buf[env_ids] *= 0.0

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    #####################################################################
    ###===========================create_sim==========================###
    #####################################################################
    def create_sim(self):
        self.env_total_width, self.env_total_height = 2, 1.5  # X and Y coords
        self.field_width, self.field_height = 1.5, 1.3
        self.goal_width, self.goal_height = 0.1, 0.4
        self.walls_depth = 0.1  # on rules its 0.05
        low_bound = -gymapi.Vec3(self.env_total_width, self.env_total_height, 0.0) / 2
        high_bound = gymapi.Vec3(self.env_total_width, self.env_total_height, 2.0) / 2
        n_fields_row = int(np.sqrt(self.num_envs))

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._add_ground()
        for field_idx in range(self.num_envs):
            _field = self.gym.create_env(self.sim, low_bound, high_bound, n_fields_row)

            # self._add_ball(_field, field_idx)
            for team in [BLUE_TEAM]:
                for robot in [RED_ROBOT]:
                    self._add_robot(_field, field_idx, team, robot)

            for team in [BLUE_TEAM]:
                for robot in [RED_ROBOT]:
                    self._add_target(_field, field_idx, team, robot)

            self._add_field(_field, field_idx)

    def _add_ground(self):
        pp = gymapi.PlaneParams()
        pp.distance = 0.0
        pp.dynamic_friction = 0.4
        pp.normal = gymapi.Vec3(
            0, 0, 1
        )  # defaultgymapi.Vec3(0.000000, 1.000000, 0.000000)
        pp.restitution = 0.0
        pp.segmentation_id = 0
        pp.static_friction = 0.7
        self.gym.add_ground(self.sim, pp)

    def _add_ball(self, field, field_idx):
        options = gymapi.AssetOptions()
        options.density = 1130.0  # 0.046 kg
        radius = 0.02134
        pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, radius))
        asset = self.gym.create_sphere(self.sim, radius, options)
        ball = self.gym.create_actor(
            env=field, asset=asset, pose=pose, group=field_idx, filter=0b01, name='ball'
        )
        self.gym.set_rigid_body_color(field, ball, 0, gymapi.MESH_VISUAL, ORANGE_COLOR)

    def _add_target(self, env, field_id, team, idx):
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        root = os.path.dirname(os.path.abspath(__file__))
        tgt_asset = self.gym.load_asset(
            sim=self.sim,
            rootpath=root,
            filename='vss_robot_target.urdf',
            options=options,
        )
        body, tag_id, tag_team = 0, 1, 2
        wheel_radius = 0.024  # _z dimension
        pos_idx = idx + 1 if team == YELLOW_TEAM else -(idx + 1)
        pose = gymapi.Transform(p=gymapi.Vec3(0.1 * pos_idx, 0.0, -1.0))
        robot = self.gym.create_actor(
            env=env,
            asset=tgt_asset,
            pose=pose,
            group=-1,
            filter=0b111,
            name='target',
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_id, gymapi.MESH_VISUAL, TEAM_COLORS[team]
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_team, gymapi.MESH_VISUAL, ID_COLORS[idx]
        )
        props = self.gym.get_actor_rigid_shape_properties(env, robot)
        props[body].filter = 0b1111
        self.gym.set_actor_rigid_shape_properties(env, robot, props)

    def _add_robot(self, env, field_id, team, idx):
        options = gymapi.AssetOptions()
        root = os.path.dirname(os.path.abspath(__file__))
        rbt_asset = self.gym.load_asset(
            sim=self.sim,
            rootpath=root,
            filename='vss_robot.urdf',
            options=options,
        )
        body, left_wheel, right_wheel, tag_id, tag_team = 0, 1, 2, 3, 4
        wheel_radius = 0.024  # _z dimension
        pos_idx = idx + 1 if team == YELLOW_TEAM else -(idx + 1)
        pose = gymapi.Transform(p=gymapi.Vec3(0.1 * pos_idx, 0.0, wheel_radius+0.005))
        robot = self.gym.create_actor(
            env=env,
            asset=rbt_asset,
            pose=pose,
            group=field_id,
            filter=0b01,
            name='robot',
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_id, gymapi.MESH_VISUAL, TEAM_COLORS[team]
        )
        self.gym.set_rigid_body_color(
            env, robot, tag_team, gymapi.MESH_VISUAL, ID_COLORS[idx]
        )
        props = self.gym.get_actor_rigid_shape_properties(env, robot)
        props[body].friction = 0.0
        props[body].filter = 0b0
        props[left_wheel].filter = 0b11
        props[left_wheel].friction = 0.7
        props[right_wheel].filter = 0b11
        props[right_wheel].friction = 0.7
        self.gym.set_actor_rigid_shape_properties(env, robot, props)

        props = self.gym.get_actor_dof_properties(env, robot)
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.01)
        props['armature'].fill(0.0002)
        props['friction'].fill(0.0002)
        props['velocity'].fill(self.robot_max_wheel_rad_s)
        self.gym.set_actor_dof_properties(env, robot, props)

    def _add_field(self, env, env_id):
        # Using procedural assets because with an urdf file rigid contacts were not being drawn
        # _width (x), _width (_y), Depth (_z)
        total_width, total_height = self.env_total_width, self.env_total_height
        field_width, field_height = self.field_width, self.field_height
        goal_width, goal_height = self.goal_width, self.goal_height
        walls_depth = self.walls_depth

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        color = gymapi.Vec3(0.2, 0.2, 0.2)

        # Side Walls (sw)
        def add_side_walls():
            sw_width = total_width
            sw_height = (total_height - field_height) / 2
            sw_x = 0
            sw_y = (field_height + sw_height) / 2
            sw_z = walls_depth / 2
            swDirs = [(1, 1), (1, -1)]  # Top and Bottom

            sw_asset = self.gym.create_box(
                self.sim, sw_width, sw_height, walls_depth, options
            )

            for dir_x, dir_y in swDirs:
                swPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * sw_x, dir_y * sw_y, sw_z)
                )
                swActor = self.gym.create_actor(
                    env=env, asset=sw_asset, pose=swPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, swActor, 0, gymapi.MESH_VISUAL, color
                )

        # End Walls (ew)
        def add_end_walls():
            ew_width = (total_width - field_width) / 2
            ew_height = (field_height - goal_height) / 2
            ew_x = (field_width + ew_width) / 2
            ew_y = (field_height - ew_height) / 2
            ew_z = walls_depth / 2
            ewDirs = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # Corners

            ew_asset = self.gym.create_box(
                self.sim, ew_width, ew_height, walls_depth, options
            )

            for dir_x, dir_y in ewDirs:
                ewPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * ew_x, dir_y * ew_y, ew_z)
                )
                ewActor = self.gym.create_actor(
                    env=env, asset=ew_asset, pose=ewPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, ewActor, 0, gymapi.MESH_VISUAL, color
                )

        # Goal Walls (gw)
        def add_goal_walls():
            gw_width = ((total_width - field_width) / 2) - goal_width
            gw_height = goal_height
            gw_x = (total_width - gw_width) / 2
            gw_y = 0
            gw_z = walls_depth / 2
            gwDirs = [(-1, 1), (1, 1)]  # left and right

            gw_asset = self.gym.create_box(
                self.sim, gw_width, gw_height, walls_depth, options
            )

            for dir_x, dir_y in gwDirs:
                gwPose = gymapi.Transform(
                    p=gymapi.Vec3(dir_x * gw_x, dir_y * gw_y, gw_z)
                )
                gwActor = self.gym.create_actor(
                    env=env, asset=gw_asset, pose=gwPose, group=env_id, filter=0b10
                )
                self.gym.set_rigid_body_color(
                    env, gwActor, 0, gymapi.MESH_VISUAL, color
                )

        add_side_walls()
        add_end_walls()
        add_goal_walls()


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_goto_obs(tgt_pos, r_pos, r_vel, r_quats, r_w, r_acts):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    rbt_angles = get_euler_xyz(r_quats.squeeze())[2].view(-1, 1, 1)
    return torch.cat(
        (
            tgt_pos,
            r_pos,
            r_vel,
            torch.cos(rbt_angles),
            torch.sin(rbt_angles),
            r_w,
            r_acts
        ), dim=-1
    ).squeeze()

@torch.jit.script
def compute_goto_obs_w_angles(tgt_pos, tgt_quats, r_pos, r_vel, r_quats, r_w, r_acts):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    tgt_angles = get_euler_xyz(tgt_quats.squeeze())[2].view(-1, 1, 1)
    rbt_angles = get_euler_xyz(r_quats.squeeze())[2].view(-1, 1, 1)
    return torch.cat(
        (
            tgt_pos,
            torch.cos(tgt_angles),
            torch.sin(tgt_angles),
            r_pos,
            r_vel,
            torch.cos(rbt_angles),
            torch.sin(rbt_angles),
            r_w,
            r_acts
        ), dim=-1
    ).squeeze()

@torch.jit.script
def calculate_distance(pos0, pos1):
    # type: (Tensor, Tensor) -> Tensor
    return torch.norm(pos0 - pos1, dim=-1).squeeze()

@torch.jit.script
def calculate_angle_diff(quat0, quat1):
    # type: (Tensor, Tensor) -> Tensor
    rad_diff = quat_diff_rad(quat0.squeeze(), quat1.squeeze())
    return torch.where(
            rad_diff > (torch.pi/2),
            torch.pi - rad_diff,
            rad_diff
        )
        
@torch.jit.script
def check_terminal_state(
    reset_buf,
    tgt_dist,
    angle_diff,
    robot_speed,
    dist_threshold,
    angle_threshold,
    speed_threshold
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tensor
    return torch.where(
            (tgt_dist < dist_threshold)
            & (angle_diff < angle_threshold)
            & (robot_speed < speed_threshold),
            torch.ones_like(reset_buf),
            torch.zeros_like(reset_buf),
        )

@torch.jit.script
def compute_goto_move_rew(p_robots, robots, targets):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    # Previous Distance
    p_dist = torch.norm(p_robots - targets, dim=-1)

    # Current Distance
    dist = torch.norm(robots - targets, dim=-1)

    return (p_dist - dist), dist

@torch.jit.script
def compute_goto_dones(
    terminal,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, Tensor, float) -> Tensor


    return torch.where(
        progress_buf >= max_episode_length,
        torch.ones_like(terminal),
        terminal
        )
