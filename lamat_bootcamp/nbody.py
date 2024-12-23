"""
This is a quick demo of a 2-body (or 3-body?) solver using only base numpy and matplotlib.
"""
# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from tqdm import tqdm, trange

G = 1.184e-4 # universal gravitational constant in AU-Mearth-year
Mearth = 1.0
Msun = 333000.0 # Earth masses
vearth = 6.283 # AU / year
AU = 1.0
year = 1.0

# This defines a gravitating body so that we can have variables "belong to" a particular body
# like the position of the Earth.
# You're welcome to ask me about this later, but you won't need to know it.
@dataclass
class Body:
    """
    The dynamical state of a gravitating body.
    """
    position: np.array
    velocity: np.array
    acceleration: np.array
    mass: float
    
def gravitational_force(body1, body2):
    """
    Computes the directed gravitational force between body1 and body2.
    """
    displacement_vector = body1.position - body2.position
    distance = np.sqrt(np.sum(displacement_vector ** 2))
    return G * body1.mass * body2.mass * displacement_vector / distance ** 3

def update_body(body, new_acceleration, timestep):
    """
    Updates the state of a body due to its acceleration.
    """
    new_position = body.position + body.velocity * timestep + body.acceleration * (timestep ** 2) / 2
    new_velocity = body.velocity + (body.acceleration + new_acceleration) * timestep / 2
    body.position = new_position
    body.velocity = new_velocity
    body.acceleration = new_acceleration

def step_forward_2(bodies, timestep):
    """
    Updates the state of body1 and body2 due to their gravitational interaction.
    """
    body1, body2 = bodies
    grav = gravitational_force(body1, body2)
    accels = [-grav / body1.mass,  grav / body2.mass]
    for (a, b) in zip(accels, bodies):
        update_body(b, a, timestep)

def step_forward_3(bodies, timestep):
    """
    How do you think I should fill this in?
    """
    pass

def evolve(bodies, step, timestep, nsteps):
    """
    Evolves the nbody system for time (timestep * nsteps) by repeatedly calling "step".
    """
    positions = [[] for _ in bodies]
    for _ in trange(nsteps):
        step(bodies, timestep)
        for (i, b) in enumerate(bodies):
            positions[i].append(np.copy(b.position))
    return np.array(positions)

def plot_nbody_frames(positions):
    n = positions.shape[1]
    for i in range(0, n, n // 10):
        pos = positions[:,i,:]
        for p in pos:
            plt.scatter([p[0]], p[1])
        plt.gca().set_xlim(1.05 * np.min(positions[:,:,0]), 1.05 * np.max(positions[:,:,0]))
        plt.gca().set_ylim(1.05 * np.min(positions[:,:,1]), 1.05 * np.max(positions[:,:,1]))
        plt.gca().set_aspect('equal')
        plt.show()

    
# %%
earth = Body(np.array([AU, 0.0]), np.array([0.0, vearth]), np.array([0.0, 0.0]), Mearth)
sun = Body(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), Msun)
positions = evolve([earth, sun], step_forward_2, year / 36000, 72000)
plot_nbody_frames(positions)