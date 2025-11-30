import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tqdm

# ======================================================
# 1. COST MAP
# ======================================================
H, W = 120, 200
cost_map_np = np.zeros((1, 1, H, W))

cy, cx = 60, 120
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
dist = np.sqrt((X-cx)**2 + (Y-cy)**2)

cost_map_np[0,0] = np.exp(-(dist**2) / (2 * 12**2)) * 10.0
cost_map_np[0,0][cost_map_np[0,0] < 2.0] = 0.0
cost_map = torch.tensor(cost_map_np, dtype=torch.float32)


def sample_cost_map(cost_map, pts):
    gx = pts[:,0] / (W-1) * 2 - 1
    gy = pts[:,1] / (H-1) * 2 - 1
    grid = torch.stack([gx, gy], dim=1).view(1,-1,1,2)
    sampled = F.grid_sample(cost_map, grid, align_corners=True)
    return sampled.view(-1)


# ======================================================
# 2. BICYCLE MODEL
# ======================================================
def rollout_bicycle(x0, u, dt=0.1, L=2.5, noise_std=0.0):
    xs = [x0]
    x = x0

    for steer, accel in u:
        if noise_std > 0:
            steer = steer + noise_std * torch.randn_like(steer)
            accel = accel + noise_std * torch.randn_like(accel)

        nx = torch.stack([
            x[0] + x[3] * torch.cos(x[2]) * dt,
            x[1] + x[3] * torch.sin(x[2]) * dt,
            x[2] + x[3] / L * torch.tan(steer) * dt,
            x[3] + accel * dt
        ])
        x = nx
        xs.append(x)

    return torch.stack(xs)


# ======================================================
# 3. COST FUNCTION
# ======================================================
def compute_cost(xs, u, goal):
    OBSTACLE_WEIGHT = 200.0
    cost = 0.0

    cost += 5.0 * torch.sum((xs[-1,:2] - goal)**2)

    pts = xs[:, :2]
    cost += OBSTACLE_WEIGHT * torch.sum(sample_cost_map(cost_map, pts))

    du = u[1:] - u[:-1]
    cost += 0.1 * torch.sum(du**2)

    cost += 0.05 * torch.sum(u[:,0]**2)

    cost += 0.1 * torch.sum((xs[:,3] - 5.0)**2)

    return cost


# ======================================================
# 4. MPC LOOP (RUNS FIRST, STORES TRAJECTORY + LOSSES)
# ======================================================
T = 40
N_STEPS = 80
DT = 0.1

x = torch.tensor([20.0, 60.0, 0.0, 0.0])
goal = torch.tensor([180.0, 60.0])

trajectory = [x.detach().numpy()]
horizon_history = []
loss_history = []   # list of [loss over iterations per MPC tick]

for step in tqdm.tqdm(range(N_STEPS)):
    # controls for this MPC cycle
    u = torch.zeros(T, 2, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=0.05)

    losses_this_tick = []

    for it in range(120):
        optimizer.zero_grad()
        xs = rollout_bicycle(x, u)
        cost = compute_cost(xs, u, goal)
        cost.backward()
        optimizer.step()

        losses_this_tick.append(cost.item())

    loss_history.append(losses_this_tick)

    # Save horizon for animation
    horizon_history.append(xs.detach().numpy())

    # Execute first control with noise
    u0 = u[0].detach()
    steer = u0[0] + 0.05 * torch.randn(())
    accel = u0[1] + 0.1 * torch.randn(())

    x = rollout_bicycle(x, u0.view(1,2), noise_std=0.02)[-1]

    trajectory.append(x.detach().numpy())

    if torch.norm(x[:2] - goal) < 3.0:
        break

trajectory = np.stack(trajectory)


# ======================================================
# 5. ANIMATION: TWO SUBPLOTS
# ======================================================
fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(14,6))

# --- Left: MPC Trajectory ---
ax_map.imshow(cost_map_np[0,0], cmap='viridis', origin='lower')
line_traj, = ax_map.plot([], [], 'r-', linewidth=2)
line_horizon, = ax_map.plot([], [], 'w--', linewidth=1)
point_car, = ax_map.plot([], [], 'ro', markersize=6)
ax_map.scatter([goal[0]],[goal[1]], c='red', s=60)
ax_map.set_title("MPC Trajectory Rollout")

# --- Right: Loss Curve per MPC Tick ---
ax_loss.set_xlim(0, 120)
ax_loss.set_ylim(0, max(max(l) for l in loss_history)*1.1)
loss_line, = ax_loss.plot([], [], '-b')
ax_loss.set_title("Optimization Loss per MPC Tick")
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Loss")

def animate(i):

    # trajectory up to now
    line_traj.set_data(trajectory[:i,0], trajectory[:i,1])

    # current car position
    point_car.set_data([trajectory[i,0]], [trajectory[i,1]])

    # current optimized horizon
    if i < len(horizon_history):
        h = horizon_history[i]
        line_horizon.set_data(h[:,0], h[:,1])

    # loss curve for current tick
    if i < len(loss_history):
        losses = loss_history[i]
        loss_line.set_data(range(len(losses)), losses)

    return line_traj, line_horizon, point_car, loss_line


ani = animation.FuncAnimation(
    fig, animate, frames=len(trajectory), interval=120, blit=False
)

plt.show()
