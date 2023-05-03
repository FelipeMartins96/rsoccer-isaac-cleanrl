import os
import torch
import numpy as np
from torch import Tensor
from gym.spaces import Box
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis, get_euler_xyz
from isaacgymenvs.tasks.base.vec_task import VecTask
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
        self.w_goal = cfg['env']['rew_weights']['goal']
        self.w_grad = cfg['env']['rew_weights']['grad']
        self.w_move = cfg['env']['rew_weights']['move']
        self.w_energy = cfg['env']['rew_weights']['energy']
        self.robot_max_wheel_rad_s = 42.0
        self.min_robot_placement_dist = 0.07
        self.cfg = cfg
        self.envs = []
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
        self.target_geom = gymutil.WireframeSphereGeometry(0.04)

    def _acquire_tensors(self):
        """Acquire and wrap tensors. Create views."""
        n_field_actors = 8  # 2 side walls, 4 end walls, 2 goal walls
        total_robots = NUM_ROBOTS * NUM_TEAMS
        num_actors = total_robots + n_field_actors
        self.s_robots = slice(0, total_robots)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state).view(
            self.num_envs, num_actors, 13
        )

        self.root_pos = self.root_state[..., 0:2]
        self.robots_pos = self.root_pos[:, self.s_robots, :]

        self.root_quats = self.root_state[..., 3:7]
        self.robots_quats = self.root_quats[:, self.s_robots, :]
        self.root_vel = self.root_state[..., 7:9]
        self.robots_vel = self.root_vel[:, self.s_robots, :]
        self.root_ang_vel = self.root_state[..., 12]
        self.robots_ang_vel = self.root_ang_vel[:, self.s_robots]

        self._refresh_tensors()
        self.env_reset_root_state = self.root_state[0].clone()

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
        self.targets = torch.zeros_like(self.robots_pos)

    #####################################################################
    ###==============================step=============================###
    #####################################################################
    def pre_physics_step(self, _actions):
        # reset progress_buf for envs reseted on previous step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.progress_buf[env_ids] = 0
        self.dof_velocity_buf[:] = _actions.to(self.device)

        act = self.dof_velocity_buf * self.robot_max_wheel_rad_s
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(act))

        self.gym.clear_lines(self.viewer)
        for idx, _env in enumerate(self.envs):
            gymutil.draw_lines(
                self.target_geom, 
                self.gym, 
                self.viewer, 
                _env, 
                gymapi.Transform(p=gymapi.Vec3(self.targets[idx,0,0], self.targets[idx,0,1], 0.0))
            )

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
        # TODO
        # self.obs_buf[:] = compute_obs(
        #     self.ball_pos,
        #     self.ball_vel,
        #     self.robots_pos,
        #     self.robots_vel,
        #     self.robots_quats,
        #     self.robots_ang_vel,
        #     self.dof_velocity_buf,
        #     self.permutations,
        #     self.mirror_tensor,
        # )
        pass

    def compute_rewards_and_dones(self):
        return
        prev_ball_pos = self.ball_pos.clone()
        prev_robots_pos = self.robots_pos.clone()

        self._refresh_tensors()
        self.rew_buf *= 0
        # Goal Rewards
        if self.w_goal > 0:
            goal_rew = (
                compute_goal_rew(
                    self.reset_buf, self.ball_pos, self.field_width, self.goal_height
                )
                * self.w_goal
            )
            self.rew_buf[..., 0] = goal_rew
        # Grad Rewards
        if self.w_grad > 0:
            grad_rew = (
                compute_grad_rew(prev_ball_pos, self.ball_pos, self.yellow_goal)
                * self.w_grad
            )
            self.rew_buf[..., 1] = grad_rew
        # Move Rewards
        if self.w_move > 0:
            move_rew = (
                compute_move_rew(
                    prev_robots_pos,
                    self.robots_pos,
                    prev_ball_pos,
                    self.ball_pos,
                )
                * self.w_move
            )
            self.rew_buf[..., 2] += move_rew
        # Energy Reward (Penalization)
        if self.w_energy > 0:
            energy_rew = compute_energy_rew(self.dof_velocity_buf) * self.w_energy
            self.rew_buf[..., 3] += energy_rew

        # Dones
        self.reset_buf = compute_vss_dones(
            ball_pos=self.ball_pos,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            max_episode_length=self.max_episode_length,
            field_width=self.field_width,
            goal_height=self.goal_height,
        )

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
            self.targets[env_ids] = rand_pos[:, 1:]

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

            # # randomize ball velocities
            # rand_ball_vel = (
            #     torch.rand(
            #         (len(env_ids), 2),
            #         dtype=torch.float,
            #         device=self.device,
            #         requires_grad=False,
            #     )
            #     - 0.5
            # ) * 1

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

            self._add_field(_field, field_idx)
            self.envs.append(_field)

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
        pose = gymapi.Transform(p=gymapi.Vec3(0.1 * pos_idx, 0.0, wheel_radius))
        robot = self.gym.create_actor(
            env=env,
            asset=rbt_asset,
            pose=pose,
            group=field_id,
            filter=0b00,
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
def compute_obs(b_pos, b_vel, r_pos, r_vel, r_quats, r_w, r_acts, perms, mirror_tensor):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    mirror_tensor = torch.tensor(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dtype=torch.float,
        device="cuda:0",
        requires_grad=False,
    )
    ball = torch.cat((b_pos, b_vel), dim=-1).repeat_interleave(3, 0).view(-1, 1, 1, 4)
    angles = get_euler_xyz(r_quats.reshape(-1, 4))[2].view(-1, 1, 3, 1)
    robots = torch.cat(
        (
            r_pos,
            r_vel,
            torch.cos(angles),
            torch.sin(angles),
            r_w,
            r_acts,
        ),
        dim=-1,
    )
    obs = torch.cat(
        (
            ball,
            robots[:, 0, perms].view(-1, 1, 3, 27),
            robots[:, 1, :, :-2].repeat_interleave(3, 0).view(-1, 1, 3, 21),
        ),
        -1,
    )
    robots *= mirror_tensor
    obs = torch.cat(
        (
            obs,
            torch.cat(
                (
                    -ball,
                    robots[:, 1, perms].view(-1, 1, 3, 27),
                    robots[:, 0, :, :-2].repeat_interleave(3, 0).view(-1, 1, 3, 21),
                ),
                -1,
            ),
        ),
        1,
    )
    return obs


@torch.jit.script
def compute_goal_rew(reset_buf, ball_pos, field_width, goal_height):
    # type: (Tensor, Tensor, float, float) -> Tensor
    ones = torch.ones_like(reset_buf)
    zeros = torch.zeros_like(ones)

    # CHECK GOAL
    is_goal = (torch.abs(ball_pos[:, 0]) > (field_width / 2)) & (
        torch.abs(ball_pos[:, 1]) < (goal_height / 2)
    )
    is_goal_blue = is_goal & (ball_pos[..., 0] > 0)
    is_goal_yellow = is_goal & (ball_pos[..., 0] < 0)

    goal = torch.where(is_goal_blue, ones, zeros)
    goal = torch.where(is_goal_yellow, -ones, goal)
    goal = goal.view(-1, 1, 1).expand(-1, 1, 3)
    return torch.cat((goal, -goal), 1)


@torch.jit.script
def compute_grad_rew(prev_ball_pos, ball_pos, yellow_goal):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    # Prev Pot
    dist_ball_left_goal = torch.norm(prev_ball_pos - (-yellow_goal), dim=1)
    dist_ball_right_goal = torch.norm(prev_ball_pos - yellow_goal, dim=1)
    prev_pot = dist_ball_left_goal - dist_ball_right_goal

    # New Pot
    dist_ball_left_goal = torch.norm(ball_pos - (-yellow_goal), dim=1)
    dist_ball_right_goal = torch.norm(ball_pos - yellow_goal, dim=1)
    pot = dist_ball_left_goal - dist_ball_right_goal

    grad = (pot - prev_pot).view(-1, 1, 1).expand(-1, 1, 3)
    return torch.cat((grad, -grad), 1)


@torch.jit.script
def compute_move_rew(p_robots, robots, p_ball, ball):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    # Previous Distance
    p_dist = torch.norm(p_robots.view(-1, 6, 2) - p_ball.unsqueeze(1), dim=-1)

    # Current Distance
    dist = torch.norm(robots.view(-1, 6, 2) - ball.unsqueeze(1), dim=-1)

    return (p_dist - dist).view(-1, 2, 3)


@torch.jit.script
def compute_energy_rew(actions):
    # type: (Tensor) -> Tensor
    return -torch.mean(torch.abs(actions), dim=-1)


@torch.jit.script
def compute_vss_dones(
    ball_pos,
    reset_buf,
    progress_buf,
    max_episode_length,
    field_width,
    goal_height,
):
    # type: (Tensor, Tensor, Tensor, float, float, float) -> Tensor

    # CHECK GOAL
    is_goal = (torch.abs(ball_pos[:, 0]) > (field_width / 2)) & (
        torch.abs(ball_pos[:, 1]) < (goal_height / 2)
    )
    ones = torch.ones_like(reset_buf)
    reset = torch.zeros_like(reset_buf)

    reset = torch.where(is_goal, ones, reset)
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)

    return reset
