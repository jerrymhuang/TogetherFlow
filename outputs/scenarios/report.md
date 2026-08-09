# Navigation scenarios

Behavioural checks on the generative model, run at fixed parameters. Two agents are counted as colliding below a body diameter of 0.4 room units. `published` is the model without a separation term; `separation` adds it at rho=0.4, kappa=0.6, and `separation (strong)` at rho=0.6, kappa=1.5 to show where the mechanism over-disperses. All three use the corrected reflecting boundary, and all beacons lie outside the room.

## Two groups crossing with opposing goals

*Addresses:* R1 point 1, R1 point 3  
*Question:* When two groups must pass through each other, do they interpenetrate as overlapping point particles, or do they resolve the encounter?

![crossing-groups](figures/crossing-groups.png)

| Metric | published | separation | separation (strong) |
|---|---|---|---|
| Collision rate (pairs < body diameter) | 0.048 | 0.033 | 0.018 |
| Minimum clearance | 0.002 | 0.001 | 0.014 |
| 5th pct nearest-neighbour clearance | 0.052 | 0.090 | 0.149 |
| Fraction of agent-steps outside the room | 0.000 | 0.000 | 0.000 |
| Order parameter (mean) | 0.333 | 0.342 | 0.349 |
| Groups swapped sides (1 = yes) | 1.000 | 1.000 | 1.000 |
| Minimum cross-group clearance | 0.053 | 0.055 | 0.040 |
| Cross-group collision rate | 0.001 | 0.001 | 0.001 |

## Group traversing a field of obstacles

*Addresses:* R1 point 3  
*Question:* Do agents route around solid obstacles, or does the model rely on the obstacle being invisible to the dynamics?

![obstacle-field](figures/obstacle-field.png)

| Metric | published | separation | separation (strong) |
|---|---|---|---|
| Collision rate (pairs < body diameter) | 0.058 | 0.053 | 0.033 |
| Minimum clearance | 0.002 | 0.006 | 0.003 |
| 5th pct nearest-neighbour clearance | 0.066 | 0.085 | 0.119 |
| Fraction of agent-steps outside the room | 0.000 | 0.000 | 0.000 |
| Order parameter (mean) | 0.387 | 0.424 | 0.335 |
| Fraction of agent-steps inside an obstacle | 0.000 | 0.000 | 0.000 |

## Group formation at the wall facing a beacon

*Addresses:* R1 point 3  
*Question:* Convergence on one region is the worst case for crowding. Does the group consolidate into a plausible cluster, or collapse into a pile?

![beacon-formation](figures/beacon-formation.png)

| Metric | published | separation | separation (strong) |
|---|---|---|---|
| Collision rate (pairs < body diameter) | 0.044 | 0.044 | 0.021 |
| Minimum clearance | 0.004 | 0.002 | 0.005 |
| 5th pct nearest-neighbour clearance | 0.059 | 0.078 | 0.118 |
| Fraction of agent-steps outside the room | 0.000 | 0.000 | 0.000 |
| Order parameter (mean) | 0.215 | 0.350 | 0.223 |

## Group exiting a room through a single door

*Addresses:* R1 point 3  
*Question:* Does the group form a queue at a door narrower than itself, and does everyone actually get out?

![room-exit](figures/room-exit.png)

| Metric | published | separation | separation (strong) |
|---|---|---|---|
| Collision rate (pairs < body diameter) | 0.065 | 0.048 | 0.019 |
| Minimum clearance | 0.003 | 0.008 | 0.009 |
| 5th pct nearest-neighbour clearance | 0.070 | 0.088 | 0.144 |
| Fraction of agent-steps outside the room | 0.786 | 0.774 | 0.690 |
| Order parameter (mean) | 0.448 | 0.379 | 0.293 |
| Fraction exited by the final step | 0.964 | 0.857 | 0.821 |
| Median exit time (s) | 8.800 | 9.250 | 8.850 |
