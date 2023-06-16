# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import OrderedDict

import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-june-net",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")    
    parser.add_argument("--wandb-notes", type=str, default=None,
        help="run notes")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="sa",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0009,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=3600,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=300,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--adaptative-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle adaptative learning rate for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=3,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=7,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.005,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=4,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=1.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--threshold-kl", type=float, default=0.008,
        help="the target KL threshold for adaptative learning rate")

    parser.add_argument("--reward-scaler", type=float, default=1000,
        help="the scale factor applied to the reward during training")
    parser.add_argument("--record-video-step-frequency", type=int, default=12000,
        help="the frequency at which to record the videos")
    parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle test runs, save to test path.")
    
    parser.add_argument("--terminal-rw", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--check-angle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--check-speed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--hierarchical", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--no-move", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--no-energy", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--speed-factor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--speed-factor-end", type=float, default=0.5)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_acts):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_acts), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_acts))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        args.wandb_project_name = "test-cleanrl-rsoccer"
        args.total_timesteps = 1000000

    run_name = f"{args.exp_name}-{args.env_id}"
    save_path = f"runs/{args.exp_name}/{run_name}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=run_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            notes=args.wandb_notes,
        )
    writer = SummaryWriter(f"{save_path}/{wandb.run.id if args.track else 'non-tracked'}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.env_id != 'goto':
        from envs.wrappers import make_env
        from envs.wrappers import RecordEpisodeStatisticsTorchVSS as RecordEpisodeStatisticsTorch
        unwrapped_env, envs = make_env(args)
    else:
        from envs.wrappers import make_env_goto
        from envs.wrappers import RecordEpisodeStatisticsTorch
        unwrapped_env = None
        envs = make_env_goto(args)

    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"{save_path}/{wandb.run.id if args.track else ''}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=400,  # for each video record up to 100 steps
        )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        n_obs=np.array(envs.single_observation_space.shape).prod(),
        n_acts=np.prod(envs.single_action_space.shape),
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    next_timeouts = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    next_values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        if args.speed_factor:
            envs.set_speed_factor(min(((update - 1.0) / num_updates) / args.speed_factor_end, 1.0))

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            next_dones[step] = next_done
            next_timeouts[step] = info["time_outs"]
            with torch.no_grad():
                next_values[step] = agent.get_value(info['terminal_observation']).flatten()
            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        if args.env_id != 'goto':
                            for rw_key in info['r']:
                                writer.add_scalar(f"rws/episodic_{rw_key}", info['r'][rw_key][idx].item(), global_step)
                        else:
                            writer.add_scalar("rws/episodic_reward", info["r"][idx], global_step)
                            for rw_key in info['log']:
                                writer.add_scalar(f"error/{rw_key}", info['log'][rw_key][idx].item(), global_step)
                        writer.add_scalar("rws/episodic_length", info["l"][idx], global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            next_non_terminal = 1.0 - next_dones.logical_and(next_timeouts.logical_not()).int()
            for t in reversed(range(args.num_steps)):
                # if t == args.num_steps - 1:
                #     nextnonterminal = 1.0 - next_done
                #     nextvalues = next_value
                # else:
                #     nextnonterminal = 1.0 - dones[t + 1]
                #     nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * next_values[t] * next_non_terminal[t] - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * (1.0 - next_dones[t]) * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for mb_i in range(args.num_minibatches):
                start = mb_i * args.minibatch_size
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.adaptative_lr:
                    current_lr = optimizer.param_groups[0]["lr"]
                    if approx_kl > (2.0 * args.threshold_kl):
                        optimizer.param_groups[0]["lr"] = max(current_lr / 1.5, 1e-6)
                    elif approx_kl < (0.5 * args.threshold_kl):
                        optimizer.param_groups[0]["lr"] = min(current_lr * 1.5,1e-2)

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("losses/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if args.env_id != 'goto':
            writer.add_scalar("losses/speed_factor", envs.speed_factor, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/returns", returns.mean().item(), global_step)
        writer.add_scalar("losses/advantages", advantages.mean().item(), global_step)
        writer.add_scalar("losses/values", values.mean().item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("Charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("Charts/update", update, global_step)

    # Save Model
    if args.track:
        save_dict = OrderedDict([
            ("env_id", args.env_id),
            ("state_dict", agent.state_dict()),
            ('n_obs', np.array(envs.single_observation_space.shape).prod()),
            ('n_acts', np.prod(envs.single_action_space.shape)),
            ('run_id', wandb.run.id),
        ])
        torch.save(save_dict, f"{save_path}/agent-{wandb.run.id}.pt")
        wandb.save(f"{save_path}/agent-{wandb.run.id}.pt")
