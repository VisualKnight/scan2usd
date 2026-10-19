"""Train the hand-place policy: behavior cloning on the scripted teacher, then DAgger (Isaac Lab, GPU PhysX).

    set OMNI_KIT_ACCEPT_EULA=YES
    D:\\il\\venv\\Scripts\\python.exe scripts\\rl\\train_policy.py [--num_envs 512] [--rounds 8] [--power_w 400]

Round 0: the teacher drives every cell (demonstrations); the policy is fit to its targets (policy.py).
Rounds 1..: the policy drives a growing share of the cells (the rest stay with the teacher, which keeps the
data on the teacher's path too), the teacher labels every state visited, the data is aggregated and the
policy is fine-tuned. Every round logs the policy's own success on the cells it drove. A final evaluation
runs fresh scenes (other seeds) with the policy alone. The GPU's average power is held under --power_w
(policy.PowerGovernor). Writes output/rl/policy/{policy.pt, round_<k>.pt, train_log.json}.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--demo_episodes", type=int, default=2)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--eval_envs", type=int, default=400)
parser.add_argument("--power_w", type=float, default=400.0)
parser.add_argument("--steps0", type=int, default=6000, help="gradient steps for behavior cloning")
parser.add_argument("--steps", type=int, default=3000, help="gradient steps per DAgger round")
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out", default=None)
parser.add_argument("--obs_set", default="scene", help="policy.OBS_SETS")
parser.add_argument("--dart", default="", help="teacher cells' executed-target noise (m, rad, joint rad)")
parser.add_argument("--obs_noise", type=float, default=0.05, help="input noise in training (in units of each input's std)")
parser.add_argument("--resume", default="", help="a checkpoint to start from (skips the demos' BC)")
parser.add_argument("--demos_file", default="", help="demonstrations saved by an earlier run (demos.pt): no re-simulation")
parser.add_argument("--max_hours", type=float, default=5.0, help="stop starting new rounds after this")
parser.add_argument("--snapshots", default="", help="comma-separated BC steps: save the network part-way (bc_<steps>.pt)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hand_teacher as T  # noqa: E402
import policy as PL  # noqa: E402
from hand_place_env import HandPlaceEnv, HandPlaceEnvCfg, SPEC  # noqa: E402

OUT = args.out or os.path.join(T.ROOT, "output", "rl", "policy")
LOG = os.path.join(OUT, "train_log.json")
DART = tuple(float(v) for v in args.dart.split(",")) if args.dart else None


def make_env(n):
    cfg = HandPlaceEnvCfg()
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = 1.0
    cfg.episode_length_s = 1.0e5  # episodes are run by the Runner, never reset by the env
    env = HandPlaceEnv(cfg)
    env.reset()
    return env


def scenes(seed, n):
    return [SPEC.scene(np.random.default_rng([seed, i])) for i in range(n)]


class Data:
    """Aggregated (observation, label) pairs on the CPU."""

    def __init__(self):
        self.obs, self.act = [], []

    def add(self, data, mask=None):
        o, a, ok = data  # (n, steps, dim), (n,) cells whose data are finite
        keep = ok if mask is None else ok & mask.cpu()
        o, a = o[keep], a[keep]
        self.obs.append(o.reshape(-1, o.shape[-1]))
        self.act.append(a.reshape(-1, a.shape[-1]))
        return int((~ok).sum())

    def tensors(self):
        return torch.cat(self.obs), torch.cat(self.act)

    def __len__(self):
        return sum(len(o) for o in self.obs)


def fit(policy, data: Data, steps: int, dev, lr: float, log: dict, snapshots=()):
    obs, act = data.tensors()
    n = len(obs)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    policy.train()
    losses = []
    obs_scale = torch.ones(obs.shape[-1], device=dev)
    obs_scale[0] = 0.1  # the clock: barely jittered
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, n, (args.batch,))
        o, a = obs[idx].to(dev, non_blocking=True), act[idx].to(dev, non_blocking=True)
        pred = policy.forward_norm(o, args.obs_noise * torch.randn_like(o) * obs_scale)
        loss = ((pred - (a - policy.act_mean) / policy.act_std) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if s % 500 == 0:
            gov.tick()
        if s + 1 in snapshots:
            policy.eval()
            PL.save(policy, os.path.join(OUT, f"bc_{s + 1}.pt"), {"round": 0, "bc_steps": s + 1, "loss": losses[-1]})
            policy.train()
    policy.eval()
    log.update({"samples": n, "loss_first": float(np.mean(losses[:100])), "loss_last": float(np.mean(losses[-200:])),
                "fit_s": time.time() - t0})
    print(f"[train] fit {steps} steps on {n} samples: loss {log['loss_first']:.4f} -> {log['loss_last']:.4f} "
          f"({log['fit_s']:.0f} s)", flush=True)


def main():
    global gov
    os.makedirs(OUT, exist_ok=True)
    gov = PL.PowerGovernor(args.power_w)
    env = make_env(args.num_envs)
    dev = env.device
    run = PL.Runner(env, SPEC, governor=gov, obs_set=args.obs_set)
    n = args.num_envs
    t_start = time.time()
    log = {"args": vars(args), "rounds": []}
    data = Data()
    policy = PL.Policy(args.obs_set).to(dev)
    seed = 0
    if args.resume:
        policy = PL.load(args.resume, dev)
    else:
        # round 0: demonstrations (simulated, or loaded with --demos_file)
        r0 = {"round": 0, "teacher": []}
        if args.demos_file:
            o, a = torch.load(args.demos_file)
            data.obs.append(o)
            data.act.append(a)
            seed += args.demo_episodes
        for k in range(0 if args.demos_file else args.demo_episodes):
            t0 = time.time()
            res, d = run.episode(scenes(seed, n), collect=True, dart=DART)
            seed += 1
            data.add(d)
            r0["teacher"].append({**PL.summarize(res), "wall_s": time.time() - t0})
            print(f"[train] demos {k}: teacher {r0['teacher'][-1]}", flush=True)
        o, a = data.tensors()
        if not args.demos_file:
            torch.save((o, a), os.path.join(OUT, "demos.pt"))
        policy.set_stats(o, a)
        fit(policy, data, args.steps0, dev, args.lr, r0, {int(s) for s in args.snapshots.split(",") if s})
        PL.save(policy, os.path.join(OUT, "round_0.pt"), {"round": 0})
        log["rounds"].append(r0)
        PL.write_json(LOG, log)
    # DAgger
    for k in range(1, args.rounds + 1):
        if (time.time() - t_start) / 3600 > args.max_hours:
            print("[train] time budget reached", flush=True)
            break
        share = min(1.0, 0.7 + 0.1 * (k - 1))  # the policy's share of the cells
        use = torch.rand(n, generator=torch.Generator().manual_seed(1000 + k)) < share
        t0 = time.time()
        res, d = run.episode(scenes(100 + k, n), policy=policy, use_policy=use.to(dev), collect=True, dart=DART)
        data.add(d)
        rk = {"round": k, "policy_share": share, "policy": PL.summarize(res, use), "teacher": PL.summarize(res, ~use),
              "episode_s": time.time() - t0}
        print(f"[train] round {k}: policy {rk['policy']}  teacher {rk['teacher']}", flush=True)
        fit(policy, data, args.steps, dev, args.lr * 0.5, rk)
        PL.save(policy, os.path.join(OUT, f"round_{k}.pt"), {"round": k, "policy": rk["policy"]})
        rk["power"] = gov.summary()
        log["rounds"].append(rk)
        PL.write_json(LOG, log)
    PL.save(policy, os.path.join(OUT, "policy.pt"), {"rounds": len(log["rounds"]) - 1})
    # held-out evaluation: fresh scenes, the policy alone
    env_eval = env
    ev = []
    m = args.num_envs
    for k in range(max(1, int(np.ceil(args.eval_envs / m)))):
        res, _ = run.episode(scenes(5000 + k, m), policy=policy, use_policy=torch.ones(m, dtype=torch.bool), collect=False)
        ev.append(res)
    res = {key: torch.cat([r[key] for r in ev])[:args.eval_envs] for key in ev[0]}
    log["eval"] = PL.summarize(res)
    log["power"] = gov.summary()
    log["hours"] = (time.time() - t_start) / 3600
    PL.write_json(LOG, log)
    print(f"[train] held-out eval: {log['eval']}  power {log['power']}", flush=True)
    env_eval.close()


if __name__ == "__main__":
    main()
    app.close()
