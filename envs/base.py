import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis
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
NUM_TEAMS = 2
NUM_ROBOTS = 3


class BASE(VecTask):
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
        self.num_fields = cfg['env']['numEnvs']
        self.max_episode_length = cfg['env']['maxEpisodeLength']
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
            cam_pos = gymapi.Vec3(1.1, 1.49, 5)
            cam_target = gymapi.Vec3(1.1, 1.5, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self._acquire_tensors()
        self._refresh_tensors()
        self.reset_dones()
        self.compute_observations()

    def allocate_buffers(self):
        # allocate buffers
        num_fields = self.num_fields
        num_obs = self.num_obs
        device = self.device
        self.obs_buf = torch.zeros(
            (num_fields, NUM_TEAMS, NUM_ROBOTS, num_obs),
            device=device,
            dtype=torch.float,
        )
        self.states_buf = torch.zeros(
            (num_fields, NUM_TEAMS, NUM_ROBOTS, num_obs),
            device=device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(
            (num_fields, NUM_TEAMS, NUM_ROBOTS), device=device, dtype=torch.float
        )
        self.reset_buf = torch.ones(num_fields, device=device, dtype=torch.long)
        self.timeout_buf = torch.zeros(num_fields, device=device, dtype=torch.long)
        self.progress_buf = torch.zeros(num_fields, device=device, dtype=torch.long)
        self.randomize_buf = torch.zeros(num_fields, device=device, dtype=torch.long)
        self.extras = {}

    def _acquire_tensors(self):
        """Acquire and wrap tensors. Create views."""
        n_field_actors = 8  # 2 side walls, 4 end walls, 2 goal walls
        num_actors = 7 + n_field_actors  # 7 = 1 ball and 6 robots
        self.s_ball = 0
        self.s_robots = slice(1, 7)

        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state).view(
            self.num_envs, num_actors, 13
        )

        self.root_pos = self.root_state[..., 0:2]
        self.robots_pos = self.root_pos[:, self.s_robots, :].view(
            -1, NUM_TEAMS, NUM_ROBOTS, 2
        )
        self.ball_pos = self.root_pos[:, self.s_ball, :]

        self.root_quats = self.root_state[..., 3:7]
        self.robots_quats = self.root_quats[:, self.s_robots, :].view(
            -1, NUM_TEAMS, NUM_ROBOTS, 4
        )

        self.root_vel = self.root_state[..., 7:9]
        self.robots_vel = self.root_vel[:, self.s_robots, :].view(
            -1, NUM_TEAMS, NUM_ROBOTS, 2
        )
        self.ball_vel = self.root_vel[:, self.s_ball, :]

        self.root_ang_vel = self.root_state[..., 12]
        self.robots_ang_vel = self.root_ang_vel[:, self.s_robots].view(
            -1, NUM_TEAMS, NUM_ROBOTS
        )

        self._refresh_tensors()
        self.env_reset_root_state = self.root_state[0].clone()

        self.rw_goal = torch.zeros_like(
            self.rew_buf[:], device=self.device, requires_grad=False
        )
        self.rw_grad = torch.zeros_like(
            self.rew_buf[:], device=self.device, requires_grad=False
        )
        self.rw_energy = torch.zeros_like(
            self.rew_buf[:], device=self.device, requires_grad=False
        )
        self.rw_move = torch.zeros_like(
            self.rew_buf[:], device=self.device, requires_grad=False
        )
        self.dof_velocity_buf = torch.zeros(
            (self.num_fields, NUM_TEAMS, NUM_ROBOTS, 2),
            device=self.device,
            requires_grad=False,
        )
        self.mirror_tensor = torch.tensor(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
            dtype=torch.float,
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
        entities_ids = list(range(7))  # 7 = 6 Robots + 1 Ball
        self.entities_pairs = torch.tensor(
            list(combinations(entities_ids, 2)), device=self.device, requires_grad=False
        )

    #####################################################################
    ###==============================step=============================###
    #####################################################################
    def pre_physics_step(self, _actions):
        pass

    def post_physics_step(self):
        pass

    def compute_observations(self):
        pass

    def reset_dones(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            # Reset env state
            self.root_state[env_ids] = self.env_reset_root_state

            close_ids = env_ids
            rand_pos = torch.zeros(
                (len(env_ids), 7, 2),  # 7 = 6 Robots + Ball, 2 = X,Y
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            while len(close_ids):
                # randomize positions
                rand_pos[close_ids] = (
                    torch.rand(
                        (len(close_ids), 7, 2),  # 7 = 6 Robots + Ball, 2 = X,Y
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

            self.ball_pos[env_ids] = rand_pos[:, self.s_ball]
            self.robots_pos[env_ids] = rand_pos[:, self.s_robots].view(
                -1, NUM_TEAMS, NUM_ROBOTS, 2
            )

            # randomize rotations
            rand_angles = torch_rand_float(
                -np.pi,
                np.pi,
                (len(env_ids), NUM_TEAMS * NUM_ROBOTS),
                device=self.device,
            )
            self.robots_quats[env_ids] = quat_from_angle_axis(
                rand_angles, self.z_axis
            ).view(-1, 2, 3, 4)

            # randomize ball velocities
            rand_ball_vel = (
                torch.rand(
                    (len(env_ids), 2),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                - 0.5
            ) * 1
            self.ball_vel[env_ids] = rand_ball_vel

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_state)
            )

            self.rw_goal[env_ids] = 0.0
            self.rw_grad[env_ids] = 0.0
            self.rw_energy[env_ids] = 0.0
            self.rw_move[env_ids] = 0.0
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
        n_fields_row = int(np.sqrt(self.num_fields))

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._add_ground()
        for field_idx in range(self.num_fields):
            _field = self.gym.create_env(self.sim, low_bound, high_bound, n_fields_row)

            self._add_ball(_field, field_idx)
            for team in [BLUE_TEAM, YELLOW_TEAM]:
                for robot in [RED_ROBOT, GREEN_ROBOT, PINK_ROBOT]:
                    self._add_robot(_field, field_idx, team, robot)

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
