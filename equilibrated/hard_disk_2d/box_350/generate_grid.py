import numpy as np

# System parameters
L = 350.0  # Box size
r = 0.5    # Disk radius
target_eta = 0.7225  # Desired packing fraction

# Area of a single disk
disk_area = np.pi * (r ** 2)

# Hexagonal lattice spacing a for a target packing fraction:
# eta = disk_area / ( (sqrt(3)/2) * a^2 )
a = np.sqrt(disk_area / (target_eta * (np.sqrt(3) / 2.0)))
dy = a * np.sqrt(3) / 2.0  # Spacing between adjacent rows

# Number of rows/cols within box 350x350
nx = int(np.floor(L / a))
ny = int(np.floor(L / dy))

positions = []
for j in range(ny):
    y = j * dy
    x_offset = (a / 2.0) if (j % 2 == 1) else 0.0
    for i in range(nx):
        x = i * a + x_offset
        if x < L and y < L:
            positions.append((x, y, 0.0))

num_particles = len(positions)
actual_eta = (num_particles * disk_area) / (L * L)

print(f"Generated {num_particles} particles.")
print(f"Calculated 2D packing fraction: {actual_eta:.5f}")

# Write extended .xyz file format
filename = "hard_disks_eta074.xyz"
header_comment = (
    f'Lattice="{L} 0.0 0.0 0.0 {L} 0.0 0.0 0.0 0.0" '
    f'Properties=species:S:1:pos:R:3 mu=12.5 potential=hard_core box={L} pbc="T T F"'
)

with open(filename, "w") as f:
    f.write(f"{num_particles}\n")
    f.write(f"{header_comment}\n")
    for pos in positions:
        f.write(f"H      {pos[0]:14.8f}     {pos[1]:14.8f}     {pos[2]:14.8f}\n")

print(f"Saved configuration to {filename}")
