"""The verbatim 2D decomposition helpers, kept only as a parity oracle.

WHY THEY ARE HERE AND NOT IN THE PACKAGE. These seven functions were ported
byte-for-byte from the original `parallelization.py` so that
`check_decomposition.py` could diff the port against its source. That job is
real and still worth doing -- but the ENGINE never calls any of them. It calls
`decomposition.grid_up_nd`, which is 79 lines.

So 1,041 lines of the package existed solely to be compared against something
else. That is test-support code by definition, and it now lives with the test
that uses it. `gcmc/space/decomposition.py` keeps only what the engine runs.

Nothing here is maintained. If a diff against the original fails, the answer is
that the ORIGINAL changed or the oracle is misconfigured -- never that this file
should be edited to agree. Editing it would make the oracle agree with itself.

UNITS: reduced throughout, as in the original.
"""

from __future__ import annotations

import numpy as np

# The original's module-level default. Preserved because the functions below
# read it, and changing it would change what the oracle compares.
cell_length_global = 30


def configure(cell_length):
    """Set the module-global decomposition cell length (legacy knob carried
    over from the original scripts; used by the checkerboard grid builders)."""
    global cell_length_global
    cell_length_global = cell_length


def grid_up_the_system(
    system_array_xyz,
    side=200,
    cell_length=30,
    buffer=2,
    x_spacing=4,
    y_spacing=3,
    x_pos=0,
    y_pos=0,
):
    """
    Divides the system into cells so that each cell can be sent to a different processor for parallelization.

    Args:
        system_array_xyz (array): All the molecules in the system.
        side (int): Side of the box.
        cell_length (int): Lenght of each cell.
        buffer (int): Buffer zone around each cell. Buffer zone never changed during gcmc in the cell
        x_spacing (int): Spacing between cells in x direction.
        y_spacing (int): Spacing between cells in y direction.
        x_pos (int): x coordinate of the origin of the grid. Cannot be 0 or side. Randomly chosen.
        y_pos (int): y coordinate of the origin of the grid. Cannot be 0 or side. Randomly chosen.

    Returns:
        A_set (list): List of all the cells in the system. Each cell contains main molecules and border molecules and cell dims.
        B_set (list): List of all the cells in the system. Each cell contains main molecules and border molecules and cell dims.
        C_set (list): List of all the cells in the system. Each cell contains main molecules and border molecules and cell dims.
        D_set (list): List of all the cells in the system. Each cell contains main molecules and border molecules and cell dims.

        cell_dims (list): [[x_lower, x_upper, y_lower, y_upper],
                           [x_lower_split, x_upper_split, y_lower_split, y_upper_split],
                           [x_lower_border, x_upper_border, y_lower_border, y_upper_border]
                         ]

    """

    # import matplotlib.pyplot as plt

    # buffer = 2
    # cell_length = 30
    # x_spacing  = 4
    # y_spacing  = 4

    # assert no_of_discretization % 2 == 0, "no_of_discretization must be even"

    # Side length of each cell
    # step = side/no_of_discretization

    # The origin to start the grid must be carefully chosen
    # so the buffer zone is not outside the system
    # this will create serious PBC issues
    # origin = create_random_origin(side, no_of_discretization, buffer)

    if x_pos == 0 and y_pos == 0:
        origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
    else:
        origin = [x_pos, y_pos, 0]

    num_squares_x = int(side / (cell_length + x_spacing))
    num_squares_y = int(side / (cell_length + y_spacing))

    # num_squares = min(num_squares_x, num_squares_y)

    # No_of_discretization must be even
    # num_squares_x = int((num_squares_x - (num_squares_x % 2))/2)
    # num_squares_y = int((num_squares_y - (num_squares_y % 2))/2)

    # x_step = (side - (num_squares * (cell_length))/num_squares
    x_step = (side / num_squares_x) - cell_length
    y_step = (side / num_squares_y) - cell_length
    # print(num_squares, space_left)

    # Total_number_of_cells = num_squares_x * num_squares_y
    # print("Total_number_of_cells", Total_number_of_cells)
    # print("num_squares_x", num_squares_x, "num_squares_y", num_squares_y)
    # print("x_step", x_step, "y_step", y_step)

    # print('')
    ##################################
    # origin = [178, 178, 0]
    # origin = [2, 178, 0]
    # buffer = 5
    # origin = [x_pos, y_pos, 0]

    # fig, ax = plt.subplots(figsize=(20, 20))
    # ax.set_aspect('equal', 'box')

    # x1, y1 = [0, 0], [0, 200]
    # x2, y2 = [0, 200], [0, 0]
    # x3, y3 = [200, 0], [200, 200]
    # x4, y4 = [200, 200], [200, 0]
    # plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, marker = 'o', markersize=1, color = 'black')
    #################################

    # Special case. 0 will wreak havoc on the code.
    # The origin can never be (0,0) or (side, side)
    if origin[0] == 0 or origin[0] == side:
        for _ in range(10):
            origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
            if origin[0] != 0.0 and origin[0] != side:
                break

    # Special case
    if origin[1] == 0 or origin[1] == side:
        for _ in range(10):
            origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
            if origin[1] != 0.0 and origin[1] != side:
                break

    origin[2] = 0.0

    # print("origin", origin)

    # assert step >= 2*buffer, "no_of_discretization "+str(step)+" is too large for the cut_off" + " " + str(buffer)
    assert x_step >= 2, (
        "Size of cell is too large " + str(x_step) + " is too large for the cut_off"
    )
    assert y_step >= 2, (
        "Size of cell is too large " + str(y_step) + " is too large for the cut_off"
    )

    # The system is divided into 4 sets of cells
    A_set = []
    # B_set = []
    # C_set = []
    # D_set = []

    x_pos = origin[0]
    y_pos = origin[1]

    rest_of_the_molecules_mask = np.ones(len(system_array_xyz), dtype=bool)

    # print("no_of_discretization", no_of_discretization)

    for _ in range(num_squares_x):
        # x limits of the cell
        x_lower = x_pos
        x_upper = x_pos + cell_length

        # x limits of the cell with buffer
        x_lower_border = x_lower - buffer
        x_upper_border = x_upper + buffer

        # Where the middle of the cell is - i.e if the cell is split
        # If cell is not split, x lower split = x upper split
        # If cell is split, x lower split = 0 and x upper split = side
        x_lower_split = (x_lower + x_upper) / 2
        x_upper_split = (x_lower + x_upper) / 2

        if x_lower_border < 0:
            x_lower_border += side

        if x_lower < 0:
            x_lower += side
            x_lower_split = 0
            x_upper_split = side

        if x_upper_border > side:
            x_upper_border -= side

        if x_upper > side:
            x_upper -= side
            x_lower_split = 0
            x_upper_split = side

        # Note for unsplit cell x lower < x upper
        # For split cell x upper < x lower as the x upper is now on the other half of the cell due to PBC
        assert x_upper_split in (side, x_lower_split)
        assert x_lower_split == 0 or x_lower_split == x_upper_split

        assert 0 < x_lower_border < side
        assert 0 < x_upper_border < side

        y_pos = origin[1]

        for _ in range(num_squares_y):
            y_lower = y_pos
            y_upper = y_pos + cell_length

            y_lower_border = y_lower - buffer
            y_upper_border = y_upper + buffer

            y_lower_split = (y_lower + y_upper) / 2
            y_upper_split = (y_lower + y_upper) / 2

            if y_lower_border < 0:
                y_lower_border += side

            if y_lower < 0:
                y_lower += side
                y_lower_split = 0
                y_upper_split = side

            if y_upper_border > side:
                y_upper_border -= side

            if y_upper > side:
                y_upper -= side
                y_upper_split = side
                y_lower_split = 0

            assert y_upper_split in (side, y_lower_split)
            assert y_lower_split == 0 or y_lower_split == y_upper_split

            assert 0 < y_lower_border < side
            assert 0 < y_upper_border < side

            # If the cell is not split, the condition is simple
            if x_upper_split == x_lower_split:
                # Get all molecules whose x and y coordinates are within the cell
                cond_main_x = (x_lower < system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] < x_upper
                )
            # The cell is split in x direction
            else:
                # Get all molecules between 0 and x upper
                cond_main_x_left = (x_upper >= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] > x_lower_split
                )
                # Get all molecules between x lower and side
                cond_main_x_right = (x_upper_split > system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] >= x_lower
                )
                # Merge both conditions
                cond_main_x = np.logical_or(cond_main_x_left, cond_main_x_right)

            if y_upper_split == y_lower_split:
                cond_main_y = (y_lower < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper
                )
            else:
                cond_main_y_left = (y_upper >= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] > y_lower_split
                )
                cond_main_y_right = (y_upper_split > system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] >= y_lower
                )
                cond_main_y = np.logical_or(cond_main_y_left, cond_main_y_right)

            # Merge the x and y conditions
            cond_main = cond_main_x & cond_main_y

            # These are the molecules in the main box
            mol_main_box = system_array_xyz[cond_main]
            rest_of_the_molecules_mask[cond_main] = False

            # Now we will get the molecules in the border (buffer zone)
            # The border (buffer zone) is divided into 8 parts
            # Bottom left corner, bottom, bottom right corner, right,
            # top right corner, top, top left corner, left

            ####################
            # Bottom

            # The lower border is normal and the cell is unsplit
            if x_upper_split == x_lower_split:
                mask_b_x = (x_lower <= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] <= x_upper
                )
            # The lower border is split. The cell is split
            elif x_upper_split == side and x_lower_split == 0:
                # molecules between 0 and x upper
                mask_b_x_left = (system_array_xyz[:, 0] <= x_upper) & (
                    x_lower_split < system_array_xyz[:, 0]
                )
                # molecules between x lower and side
                mask_b_x_right = (system_array_xyz[:, 0] < x_upper_split) & (
                    x_lower <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_b_x = np.logical_or(mask_b_x_left, mask_b_x_right)

            # The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower:
                # Molecules between y lower border and y lower
                mask_b_y = (y_lower_border < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] <= y_lower
                )
            # The lower border is split. The cell is split
            elif y_lower < y_lower_border:
                # molecules between 0 and y lower
                mask_b_y_bottom = (system_array_xyz[:, 1] <= y_lower) & (
                    system_array_xyz[:, 1] > 0
                )
                # molecules between y lower border and side
                mask_b_y_top = (system_array_xyz[:, 1] < side) & (
                    y_lower_border <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_b_y = np.logical_or(mask_b_y_bottom, mask_b_y_top)
            else:
                raise Exception("Something is wrong")

            # Combine x and y conditiosn
            mask_b = mask_b_x & mask_b_y
            # Get the molecules in the bottom border
            mol_b = system_array_xyz[mask_b]
            # rest_of_the_molecules_mask[mask_b] = False

            ####################
            # Bottom left corner

            # The right border is normal. The cell is unsplit
            if x_lower_border < x_lower:
                # Molecules between x upper and y upper border
                mask_blc_x = (x_lower_border < system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] <= x_lower
                )
            # The right border is split. The cell is split
            elif x_lower < x_lower_border:
                mask_blc_x_left = (system_array_xyz[:, 0] <= x_lower) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_blc_x_right = (system_array_xyz[:, 0] < side) & (
                    x_lower_border <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_blc_x = np.logical_or(mask_blc_x_left, mask_blc_x_right)
            else:
                raise Exception("Something is wrong")

            # The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower:
                # Molecules between y lower border and y lower
                mask_blc_y = (y_lower_border < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] <= y_lower
                )
            # The lower border is split. The cell is split
            elif y_lower < y_lower_border:
                # molecules between 0 and y lower
                mask_blc_y_bottom = (system_array_xyz[:, 1] <= y_lower) & (
                    system_array_xyz[:, 1] > 0
                )
                # molecules between y lower border and side
                mask_blc_y_top = (system_array_xyz[:, 1] < side) & (
                    y_lower_border <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_blc_y = np.logical_or(mask_blc_y_bottom, mask_blc_y_top)
            else:
                raise Exception("Something is wrong")

            # Molecules in the bottom left corner
            mask_blc = mask_blc_x & mask_blc_y
            mol_blc = system_array_xyz[mask_blc]
            # rest_of_the_molecules_mask[mask_blc] = False

            ####################
            # Bottom right corner
            # The small square of bottom left corner is normal and is in middle of the box
            if x_upper_border > x_upper:
                mask_brc_x = (x_upper <= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] < x_upper_border
                )
            # The small square of bottom left corner is split and has teleported to left side due to PBC
            elif x_upper > x_upper_border:
                # mask_brc_x = (0 < system_array_xyz[:,0] ) & (system_array_xyz[:,0] <= x upper_border)
                mask_brc_x_left = (system_array_xyz[:, 0] <= x_upper_border) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_brc_x_right = (system_array_xyz[:, 0] < side) & (
                    x_upper <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_brc_x = np.logical_or(mask_brc_x_left, mask_brc_x_right)
            else:
                raise Exception("Something is wrong")

            # The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower:
                # Molecules between y lower border and y lower
                mask_brc_y = (y_lower_border < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] <= y_lower
                )
            # The lower border is split. The cell is split
            elif y_lower < y_lower_border:
                # molecules between 0 and y lower
                mask_brc_y_bottom = (system_array_xyz[:, 1] <= y_lower) & (
                    system_array_xyz[:, 1] > 0
                )
                # molecules between y lower border and side
                mask_brc_y_top = (system_array_xyz[:, 1] < side) & (
                    y_lower_border <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_brc_y = np.logical_or(mask_brc_y_bottom, mask_brc_y_top)
            else:
                raise Exception("Something is wrong")

            mask_brc = mask_brc_x & mask_brc_y
            mol_brc = system_array_xyz[mask_brc]
            # rest_of_the_molecules_mask[mask_brc] = False

            ####################
            # Right

            # The right border is normal. The cell is unsplit
            if x_upper < x_upper_border:
                # Molecules between x upper and y upper border
                mask_r_x = (x_upper <= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] < x_upper_border
                )
            # The right border is split. The cell is split
            elif x_upper_border < x_upper:
                mask_r_x_left = (system_array_xyz[:, 0] <= x_upper_border) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_r_x_right = (system_array_xyz[:, 0] < side) & (
                    x_upper <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_r_x = np.logical_or(mask_r_x_left, mask_r_x_right)
            else:
                raise Exception("Something is wrong")

            # The right border is normal. The cell is unsplit
            if y_upper_split == y_lower_split:
                mask_r_y = (y_lower < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper
                )
            # The right border is split. The cell is split
            elif y_upper_split == side and y_lower_split == 0:
                mask_r_y_left = (y_upper >= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] > y_lower_split
                )
                mask_r_y_right = (y_upper_split > system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] >= y_lower
                )
                mask_r_y = np.logical_or(mask_r_y_right, mask_r_y_left)

            mask_r = mask_r_x & mask_r_y
            mol_r = system_array_xyz[mask_r]
            # rest_of_the_molecules_mask[mask_r] = False

            ####################
            # Top right corner

            # The right border is normal. The cell is unsplit
            if x_upper < x_upper_border:
                # Molecules between x upper and y upper border
                mask_trc_x = (x_upper <= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] < x_upper_border
                )
            # The right border is split. The cell is split
            elif x_upper_border < x_upper:
                mask_trc_x_left = (system_array_xyz[:, 0] <= x_upper_border) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_trc_x_right = (system_array_xyz[:, 0] < side) & (
                    x_upper <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_trc_x = np.logical_or(mask_trc_x_left, mask_trc_x_right)
            else:
                raise Exception("Something is wrong")

            # The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_trc_y = (y_upper <= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper_border
                )
            # The lower border is split. The cell is split
            elif y_upper_border < y_upper:
                mask_trc_y_bottom = (system_array_xyz[:, 1] <= y_upper_border) & (
                    system_array_xyz[:, 1] > 0
                )
                mask_trc_y_top = (system_array_xyz[:, 1] < side) & (
                    y_upper <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_trc_y = np.logical_or(mask_trc_y_bottom, mask_trc_y_top)
            else:
                raise Exception("Something is wrong")

            mask_trc = mask_trc_x & mask_trc_y
            mol_trc = system_array_xyz[mask_trc]
            # rest_of_the_molecules_mask[mask_trc] = False

            ####################
            # Top

            # The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_t_y = (y_upper <= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper_border
                )
            # The lower border is split. The cell is split
            elif y_upper_border < y_upper:
                mask_t_y_bottom = (system_array_xyz[:, 1] <= y_upper_border) & (
                    system_array_xyz[:, 1] > 0
                )
                mask_t_y_top = (system_array_xyz[:, 1] < side) & (
                    y_upper <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_t_y = np.logical_or(mask_t_y_bottom, mask_t_y_top)
            else:
                raise Exception("Something is wrong")

            if x_upper_split == x_lower_split:
                mask_t_x = (x_lower < system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] < x_upper
                )
            elif x_upper_split == side and x_lower_split == 0:
                mask_t_x_left = (x_upper >= system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] > x_lower_split
                )
                mask_t_x_right = (x_upper_split > system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] >= x_lower
                )
                mask_t_x = np.logical_or(mask_t_x_left, mask_t_x_right)

            mask_t = mask_t_x & mask_t_y
            mol_t = system_array_xyz[mask_t]
            # rest_of_the_molecules_mask[mask_t] = False

            ####################
            # Top left corner

            # The right border is normal. The cell is unsplit
            if x_lower_border < x_lower:
                # Molecules between x upper and y upper border
                mask_tlc_x = (x_lower_border < system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] <= x_lower
                )
            # The right border is split. The cell is split
            elif x_lower < x_lower_border:
                mask_tlc_x_left = (system_array_xyz[:, 0] <= x_lower) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_tlc_x_right = (system_array_xyz[:, 0] < side) & (
                    x_lower_border <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_tlc_x = np.logical_or(mask_tlc_x_left, mask_tlc_x_right)
            else:
                raise Exception("Something is wrong")

            # The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_tlc_y = (y_upper <= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper_border
                )
            # The lower border is split. The cell is split
            elif y_upper_border < y_upper:
                mask_tlc_y_bottom = (system_array_xyz[:, 1] <= y_upper_border) & (
                    system_array_xyz[:, 1] > 0
                )
                mask_tlc_y_top = (system_array_xyz[:, 1] < side) & (
                    y_upper <= system_array_xyz[:, 1]
                )
                # Combine both conditions
                mask_tlc_y = np.logical_or(mask_tlc_y_bottom, mask_tlc_y_top)
            else:
                raise Exception("Something is wrong")

            mask_tlc = mask_tlc_x & mask_tlc_y
            mol_tlc = system_array_xyz[mask_tlc]
            # rest_of_the_molecules_mask[mask_tlc] = False

            #####################
            # Left

            # The right border is normal. The cell is unsplit
            if x_lower_border < x_lower:
                # Molecules between x upper and y upper border
                mask_l_x = (x_lower_border < system_array_xyz[:, 0]) & (
                    system_array_xyz[:, 0] <= x_lower
                )
            # The right border is split. The cell is split
            elif x_lower < x_lower_border:
                mask_l_x_left = (system_array_xyz[:, 0] <= x_lower) & (
                    system_array_xyz[:, 0] > 0
                )
                mask_l_x_right = (system_array_xyz[:, 0] < side) & (
                    x_lower_border <= system_array_xyz[:, 0]
                )
                # Combine both conditions
                mask_l_x = np.logical_or(mask_l_x_left, mask_l_x_right)
            else:
                raise Exception("Something is wrong")

            if y_upper_split == y_lower_split:
                mask_l_y = (y_lower < system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] < y_upper
                )
            elif y_upper_split == side and y_lower_split == 0:
                mask_l_y_left = (y_upper >= system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] > y_lower_split
                )
                mask_l_y_right = (y_upper_split > system_array_xyz[:, 1]) & (
                    system_array_xyz[:, 1] >= y_lower
                )
                mask_l_y = np.logical_or(mask_l_y_left, mask_l_y_right)

            mask_l = mask_l_x & mask_l_y
            mol_l = system_array_xyz[mask_l]
            # rest_of_the_molecules_mask[mask_l] = False

            ####################

            main_mol = mol_main_box
            border_mol = np.vstack(
                (mol_blc, mol_b, mol_brc, mol_r, mol_trc, mol_t, mol_tlc, mol_l)
            )

            # plt.scatter(mol_b[:,0], mol_b[:,1], c='b', s=100, alpha=0.15)
            # plt.scatter(mol_t[:,0], mol_t[:,1], c='r', s=100, alpha=0.15)

            cell_dims = [
                [x_lower, x_upper, y_lower, y_upper],
                [x_lower_split, x_upper_split, y_lower_split, y_upper_split],
                [x_lower_border, x_upper_border, y_lower_border, y_upper_border],
            ]

            A_set.append([main_mol, border_mol, cell_dims])

            # if   i %2 == 0 and j % 2 == 0:
            #     A_set.append([main_mol, border_mol, cell_dims])

            # elif i %2 == 0 and j %2 == 1:
            #     B_set.append([main_mol, border_mol, cell_dims])

            # elif i %2 == 1 and j %2 == 0:
            #     C_set.append([main_mol, border_mol, cell_dims])

            # elif i %2 == 1 and j % 2== 1:
            #     D_set.append([main_mol, border_mol, cell_dims])

            # else: raise Exception("Something is wrong")

            y_pos += y_step + cell_length

            if y_pos > side:
                y_pos -= side

        x_pos += x_step + cell_length
        if x_pos > side:
            x_pos -= side

    # plt.show()

    rest_of_the_molecules_xyz = system_array_xyz[rest_of_the_molecules_mask]

    return A_set, rest_of_the_molecules_xyz


def create_cavity(cell_border_dims_array, box_dims, two_dim_flag):
    """Draw a random cavity centre inside one decomposition cell, honouring
    the cell's wrap-aware borders (split intervals when the cell straddles the
    periodic seam). Transcribed from the original parallel code."""

    side_of_sheet = box_dims[0]

    [x_lower, x_upper, y_lower, y_upper] = cell_border_dims_array[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_border_dims_array[
        1
    ]

    cavity_center_x = None
    cavity_center_x_left = None
    cavity_center_x_right = None

    if x_split_left == x_split_right:
        cavity_center_x = np.random.default_rng().uniform(
            low=x_lower, high=x_upper, size=1
        )[0]

    elif x_split_left == 0 and x_split_right == side_of_sheet:
        cavity_center_x_left = np.random.default_rng().uniform(
            low=x_lower, high=x_split_right, size=1
        )[0]

        cavity_center_x_right = np.random.default_rng().uniform(
            low=x_split_left,
            high=x_upper,
            size=1,
        )[0]
    else:
        raise ValueError(x_split_left, x_split_right)

    if cavity_center_x_left is not None and cavity_center_x_right is not None:
        # If not done like this, the small sliver of space created by cutting off the cell will be sampled more

        cell_side_1 = x_upper - x_split_left
        cell_side_2 = x_split_right - x_lower

        right_pecentage = cell_side_1 / (cell_side_1 + cell_side_2)
        left_pecentage = cell_side_2 / (cell_side_1 + cell_side_2)

        # assert right_pecentage + left_pecentage == 1, str(right_pecentage) + " " + str(left_pecentage)

        cavity_center_x = np.random.choice(
            [cavity_center_x_left, cavity_center_x_right],
            p=[left_pecentage, right_pecentage],
        )

    # assert cavity_center_x !=  None

    cavity_center_y = None
    cavity_center_y_left = None
    cavity_center_y_right = None

    if y_split_left == y_split_right:
        cavity_center_y = np.random.default_rng().uniform(
            low=y_lower, high=y_upper, size=1
        )[0]

    elif y_split_left == 0 and y_split_right == side_of_sheet:
        cavity_center_y_left = np.random.default_rng().uniform(
            low=y_lower, high=y_split_right, size=1
        )[0]

        cavity_center_y_right = np.random.default_rng().uniform(
            low=y_split_left, high=y_upper, size=1
        )[0]

    else:
        raise ValueError(y_split_left, y_split_right)

    if cavity_center_y_left is not None and cavity_center_y_right is not None:
        cell_side_1 = y_upper - y_split_left
        cell_side_2 = y_split_right - y_lower

        right_pecentage = cell_side_1 / (cell_side_1 + cell_side_2)
        left_pecentage = cell_side_2 / (cell_side_1 + cell_side_2)

        # assert right_pecentage + left_pecentage == 1, str(right_pecentage) + " " + str(left_pecentage)

        cavity_center_y = np.random.choice(
            [cavity_center_y_left, cavity_center_y_right],
            p=[left_pecentage, right_pecentage],
        )

    # assert cavity_center_y !=  None

    cavity_center = np.array([cavity_center_x, cavity_center_y])

    return cavity_center


def neighbor_indices_in_cavity_parallel(
    molecules_xyz, cavity_center, cavity_radius, cell_dims, box_dims, two_dim_flag
):
    """
    Find indices of molecules inside a square cavity, with PBC that respects
    cell boundaries in parallel decomposition.

    For a non-split cell [x_lower, x_upper] the cavity wraps at cell edges
    with period = cell_length.  For a split cell (straddles global box edge)
    the cavity wraps at [0, side_of_box].

    Args:
        molecules_xyz : (N, 2) or (N, 3) active molecule coordinates.
        cavity_center : (2,) or (3,) center of the square cavity.
        cavity_radius : half-side of the square cavity.
        cell_dims     : [[x_lo, x_hi, y_lo, y_hi],
                         [x_split_l, x_split_r, y_split_l, y_split_r]]
        box_dims      : [side_of_box, side_of_box] or [s, s, s].
        two_dim_flag  : bool — skip z-axis if True.

    Returns:
        1-D int array of indices into molecules_xyz that fall inside the cavity.
    """
    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]
    side_of_box = box_dims[0]

    # ---- X axis ----
    mask_x = _axis_mask(
        molecules_xyz[:, 0],
        cavity_center[0],
        cavity_radius,
        x_lower,
        x_upper,
        x_split_left,
        x_split_right,
        side_of_box,
    )

    # ---- Y axis ----
    mask_y = _axis_mask(
        molecules_xyz[:, 1],
        cavity_center[1],
        cavity_radius,
        y_lower,
        y_upper,
        y_split_left,
        y_split_right,
        side_of_box,
    )

    # ---- Z axis ----
    if two_dim_flag:
        mask_z = np.ones(len(molecules_xyz), dtype=bool)
    else:
        # Decomposition is 2-D; cells span full z range → global PBC
        cz = cavity_center[2]
        lo, hi = cz - cavity_radius, cz + cavity_radius
        if lo < 0:
            mask_z = (molecules_xyz[:, 2] >= 0) & (molecules_xyz[:, 2] <= hi)
            mask_z |= (molecules_xyz[:, 2] >= lo + side_of_box) & (
                molecules_xyz[:, 2] <= side_of_box
            )
        elif hi > side_of_box:
            mask_z = (molecules_xyz[:, 2] >= lo) & (molecules_xyz[:, 2] <= side_of_box)
            mask_z |= (molecules_xyz[:, 2] >= 0) & (
                molecules_xyz[:, 2] <= hi - side_of_box
            )
        else:
            mask_z = (molecules_xyz[:, 2] >= lo) & (molecules_xyz[:, 2] <= hi)

    return np.nonzero(mask_x & mask_y & mask_z)[0]


def _axis_mask(
    coords, center, radius, lo_bound, hi_bound, split_left, split_right, side_of_box
):
    """
    Build a boolean mask for one axis.

    Non-split cell:  wraps at [lo_bound, hi_bound]  with period = hi_bound - lo_bound
    Split cell:      wraps at [0, side_of_box]       with period = side_of_box
    """
    lo = center - radius
    hi = center + radius

    if split_left == split_right:
        # ---- Non-split cell ----
        cell_length = hi_bound - lo_bound
        if lo < lo_bound:
            # Underflow wraps to upper end of cell
            mask = (coords >= lo_bound) & (coords <= hi)
            mask |= (coords >= lo + cell_length) & (coords <= hi_bound)
        elif hi > hi_bound:
            # Overflow wraps to lower end of cell
            mask = (coords >= lo) & (coords <= hi_bound)
            mask |= (coords >= lo_bound) & (coords <= hi - cell_length)
        else:
            mask = (coords >= lo) & (coords <= hi)

    elif split_left == 0 or split_right == side_of_box:
        # ---- Split cell: wraps at global box edges ----
        if lo < 0:
            mask = (coords >= 0) & (coords <= hi)
            mask |= (coords >= lo + side_of_box) & (coords <= side_of_box)
        elif hi > side_of_box:
            mask = (coords >= lo) & (coords <= side_of_box)
            mask |= (coords >= 0) & (coords <= hi - side_of_box)
        else:
            mask = (coords >= lo) & (coords <= hi)

    else:
        raise ValueError(
            f"Invalid split: split_left={split_left}, split_right={split_right}"
        )

    return mask


def apply_pbc_array(trial_x, trial_y, cell_dims, box_dims):
    """Clip arrays of trial coordinates to one decomposition cell's wrap-aware
    interior (the array twin of `apply_PBC_single`); trials falling outside
    the cell are the neighbouring cells' business, never this one's."""

    side_of_box = box_dims[0]

    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]

    if x_split_left == x_split_right:
        # cell_length = side_of_box for serial run
        cell_length = x_upper - x_lower
        assert np.round(cell_length, 4) == np.round(cell_length_global, 4), (
            str(cell_length) + " " + str(cell_length_global)
        )

        assert cell_length >= 0

        trial_x[trial_x <= x_lower] += cell_length
        trial_x[trial_x >= x_upper] -= cell_length

        # assert np.all(trial_x >= x_lower) and np.all(trial_x <= x_upper), str(x_upper) + " " + str(x_lower)+ "-" +str(trial_x[trial_x <= x_lower]) + "+" + str(trial_x[trial_x >= x_upper])

    # This case is not used in serial run
    elif x_split_left == 0 or x_split_right == side_of_box:
        # raise ValueError(x_split_left, x_split_right)

        cell_length = x_lower - x_upper

        assert cell_length >= 0
        assert np.round(cell_length, 4) == np.round(
            side_of_box - cell_length_global, 4
        ), str(cell_length) + " " + str(side_of_box - cell_length_global)

        trial_x[trial_x <= 0] += side_of_box
        trial_x[trial_x >= side_of_box] -= side_of_box

        mask_pbc_x_1 = (side_of_box / 2 >= trial_x) & (trial_x >= x_upper)
        trial_x[mask_pbc_x_1] += cell_length

        mask_pbc_x_2 = (side_of_box / 2 <= trial_x) & (trial_x <= x_lower)
        trial_x[mask_pbc_x_2] -= cell_length

        # The order is important. Should not be applied before cell_length pbc
        trial_x[trial_x <= 0] += side_of_box
        trial_x[trial_x >= side_of_box] -= side_of_box

    else:
        raise ValueError(x_split_left, x_split_right)

    if y_split_left == y_split_right:
        cell_length = y_upper - y_lower

        assert cell_length >= 0
        assert np.round(cell_length, 4) == np.round(cell_length_global, 4), (
            str(cell_length) + " " + str(cell_length_global)
        )

        trial_y[trial_y <= y_lower] += cell_length
        trial_y[trial_y >= y_upper] -= cell_length

        # assert np.all(trial_y >= y_lower) and np.all(trial_y <= y_upper), str(y_upper) + " " + str(y_lower) + "-" +str(trial_y[trial_y <= y_lower]) + "+" + str(trial_y[trial_y >= y_upper])

    # This case is not used in serial run
    elif y_split_left == 0 or y_split_right == side_of_box:
        # raise ValueError(y_split_left, y_split_right)

        cell_length = y_lower - y_upper

        assert cell_length >= 0
        assert np.round(cell_length, 4) == np.round(
            side_of_box - cell_length_global, 4
        ), str(cell_length) + " " + str(side_of_box - cell_length_global)

        trial_y[trial_y <= 0] += side_of_box
        trial_y[trial_y >= side_of_box] -= side_of_box

        mask_pbc_y_1 = (side_of_box / 2 >= trial_y) & (trial_y >= y_upper)
        trial_y[mask_pbc_y_1] += cell_length

        mask_pbc_y_2 = (side_of_box / 2 <= trial_y) & (trial_y <= y_lower)
        trial_y[mask_pbc_y_2] -= cell_length

        # The order is important. Should not be applied before cell_length pbc
        trial_y[trial_y <= 0] += side_of_box
        trial_y[trial_y >= side_of_box] -= side_of_box

    else:
        raise ValueError(y_split_left, y_split_right)

    # Make sure all trial molecules are inside the box
    assert np.all(trial_x >= 0) and np.all(trial_x <= side_of_box), (
        str(np.all(trial_x >= 0)) + " " + str(np.all(trial_x <= side_of_box))
    )
    assert np.all(trial_y >= 0) and np.all(trial_y <= side_of_box), (
        str(np.all(trial_y >= 0)) + " " + str(np.all(trial_y <= side_of_box))
    )

    return trial_x, trial_y


def apply_PBC_single(moved_disk_x, moved_disk_y, cell_dims, box_dims):
    """Wrap one moved particle back into its decomposition cell's wrap-aware
    interval (split at the seam when the cell straddles it); the parallel
    scheme's ownership rule is that a particle never leaves its cell within a
    window."""

    side_of_box = box_dims[0]

    [x_lower, x_upper, y_lower, y_upper] = cell_dims[0]
    # x and y coordinates of the middle of the box
    [x_split_left, x_split_right, y_split_left, y_split_right] = cell_dims[1]

    if x_split_left == x_split_right:
        # assert x_lower <= x_upper, str(x_lower) + " <= " + str(x_upper)

        cell_length = x_upper - x_lower
        # assert cell_length >= 0

        if moved_disk_x <= x_lower:
            moved_disk_x += cell_length
        if moved_disk_x >= x_upper:
            moved_disk_x -= cell_length

        # assert 0 <= moved_disk_x <= side_of_box, "x " + str(0) + " <= " + str(moved_disk_x) + " <= " + str(side_of_box) + " random_x " + str(random_disk_xyz[0])
        # assert x_lower <= moved_disk_x <= x_upper, str(x_upper) + " " + str(x_lower)+ " - " +str(moved_disk_x) + " random_x " + str(random_disk_xyz[0])

    elif x_split_left == 0 or x_split_right == side_of_box:
        # assert x_lower >= x_upper, str(x_lower) + " >= " + str(x_upper)

        cell_length = x_lower - x_upper  # (side_of_box-x_lower) + (x_upper-0)
        # assert cell_length >= 0

        if moved_disk_x <= 0:
            moved_disk_x += side_of_box
        if moved_disk_x >= side_of_box:
            moved_disk_x -= side_of_box

        if side_of_box / 2 >= moved_disk_x >= x_upper:
            moved_disk_x += cell_length
        if side_of_box / 2 <= moved_disk_x <= x_lower:
            moved_disk_x -= cell_length

        # assert 0 <= moved_disk_x <= side_of_box, "x " + str(0) + " <= " + str(moved_disk_x) + " <= " + str(side_of_box) + " random_x " + str(random_disk_xyz[0]) + " +- " + str(cell_length) + " x_lower " + str(x_lower) + " x_upper " + str(x_upper)
        # assert (0 <= moved_disk_x <= x_upper) or (x_lower <= moved_disk_x <= side_of_box), str(x_upper) + " " + str(x_lower)+ "-" +str(moved_disk_x)+ " random_x " + str(random_disk_xyz[0])

    else:
        raise ValueError(x_split_left, x_split_right)

    if y_split_left == y_split_right:
        # assert y_lower <= y_upper, str(y_lower) + " <= " + str(y_upper)

        cell_length = y_upper - y_lower
        # assert cell_length >= 0

        if moved_disk_y <= y_lower:
            moved_disk_y += cell_length
        if moved_disk_y >= y_upper:
            moved_disk_y -= cell_length

        # assert 0 <= moved_disk_y <= side_of_box, "y " + str(0) + " <= " + str(moved_disk_y) + " <= " + str(side_of_box) + " random_y " + str(random_disk_xyz[1])
        # assert y_lower <= moved_disk_y <= y_upper, str(y_upper) + " " + str(y_lower)+ "-" +str(moved_disk_y) + " random_y " + str(random_disk_xyz[1])

    elif y_split_left == 0 or y_split_right == side_of_box:
        # assert y_lower >= y_upper, str(y_lower) + " >= " + str(y_upper)

        cell_length = y_lower - y_upper  # (side_of_box-y_lower) + (y_upper-0)
        # assert cell_length >= 0

        if moved_disk_y <= 0:
            moved_disk_y += side_of_box
        if moved_disk_y >= side_of_box:
            moved_disk_y -= side_of_box

        if side_of_box / 2 >= moved_disk_y >= y_upper:
            moved_disk_y += cell_length
        if side_of_box / 2 <= moved_disk_y <= y_lower:
            moved_disk_y -= cell_length

        # assert 0 <= moved_disk_y <= side_of_box, "y " + str(0) + " <= " + str(moved_disk_y) + " <= " + str(side_of_box) + " random_y " + str(random_disk_xyz[1])  + " +- " + str(cell_length) + " y_lower " + str(y_lower) + " y_upper " + str(y_upper)
        # assert (0 <= moved_disk_y <= y_upper) or (y_lower <= moved_disk_y <= side_of_box), str(y_upper) + " " + str(y_lower)+ "-" +str(moved_disk_y) + " random_y " + str(random_disk_xyz[1])

    else:
        raise ValueError(y_split_left, y_split_right)

    ###################### Energy evaluation ############################

    # moved_disk_xyz = np.array([moved_disk_x, moved_disk_y, moved_disk_z])

    return moved_disk_x, moved_disk_y
