import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


H, W = 120, 200
cost_map_np = np.zeros((1, 1, H, W))

# Obstacle: gaussian bump
cy, cx = 60, 120
Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
dist = np.sqrt((X-cx)**2 + (Y-cy)**2)

cost_map_np[0,0] = np.exp(-(dist**2) / (2 * 12**2)) * 10.0
cost_map_np[0,0][cost_map_np[0,0]<2] = 0.0  # threshold small values
cost_map = torch.tensor(cost_map_np, dtype=torch.float32)


def sample_cost_map(cost_map, pts):
    # pts: [T,2] in (x,y)
    # convert to normalized grid coords [-1,1]
    gx = pts[:,0] / (W-1) * 2 - 1
    gy = pts[:,1] / (H-1) * 2 - 1
    grid = torch.stack([gx, gy], dim=1).view(1,-1,1,2)
    sampled = F.grid_sample(cost_map, grid, align_corners=True)
    return sampled.view(-1)


def rollout_bicycle(x0, u, dt=0.1, L=2.5):
    """
    Pure functional version: no in-place modification.
    """
    xs = [x0]
    x = x0

    for steer, accel in u:
        # compute next state without modifying x in-place
        nx = torch.stack([
            x[0] + x[3] * torch.cos(x[2]) * dt,
            x[1] + x[3] * torch.sin(x[2]) * dt,
            x[2] + x[3] / L * torch.tan(steer) * dt,
            x[3] + accel * dt
        ])

        x = nx
        xs.append(x)

    return torch.stack(xs)


def compute_cost(xs, u, goal):
    OBSTACLE_WEIGHT = 200.0
    cost = 0.0

    # --- goal ---
    cost += 5.0 * torch.sum((xs[-1,:2] - goal)**2)

    # --- cost map (obstacles) ---
    pts = xs[:, :2]  # (x,y)
    cost += OBSTACLE_WEIGHT * torch.sum(sample_cost_map(cost_map, pts))

    # --- smoothness ---
    du = u[1:] - u[:-1]
    cost += 0.1 * torch.sum(du**2)

    # --- steering magnitude ---
    cost += 0.05 * torch.sum(u[:,0]**2)

    # --- velocity regulation (prefer v ~ 5 m/s) ---
    cost += 0.1 * torch.sum((xs[:,3] - 5.0)**2)

    return cost



T = 40
u = torch.zeros(T, 2, requires_grad=True)
optimizer = torch.optim.Adam([u], lr=0.05)

x0 = torch.tensor([20.0, 60.0, 0.0, 0.0])
goal = torch.tensor([180.0, 60.0])

for it in range(2000):
    optimizer.zero_grad()
    xs = rollout_bicycle(x0, u)
    cost = compute_cost(xs, u, goal)
    cost.backward()
    optimizer.step()

    if it % 20 == 0:
        print(f"iter={it}, cost={cost.item():.2f}")


xs_np = xs.detach().numpy()
plt.figure(figsize=(12,6))
plt.imshow(cost_map_np[0,0], cmap='viridis', origin='lower')
plt.plot(xs_np[:,0], xs_np[:,1], 'r', linewidth=2)
plt.scatter([x0[0]],[x0[1]], c='white', s=80)
plt.scatter([goal[0]],[goal[1]], c='red', s=80)
plt.show()