import numpy as np
#import matplotlib.pyplot as plt

def grid_up_the_system(system_array_xyz, side=200, cell_length=30, buffer=2,
                       x_spacing=4, y_spacing=3, x_pos = 0, y_pos = 0):

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

    #import matplotlib.pyplot as plt

    #buffer = 2
    #cell_length = 30
    #x_spacing  = 4
    #y_spacing  = 4

    #assert no_of_discretization % 2 == 0, "no_of_discretization must be even"

    #Side length of each cell
    #step = side/no_of_discretization
    
    #The origin to start the grid must be carefully chosen
    #so the buffer zone is not outside the system
    #this will create serious PBC issues
    #origin = create_random_origin(side, no_of_discretization, buffer)

    if x_pos == 0 and y_pos == 0:
        origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
    else:
        origin = [x_pos, y_pos, 0]

    num_squares_x = int(side/(cell_length + x_spacing))
    num_squares_y = int(side/(cell_length + y_spacing))

    #num_squares = min(num_squares_x, num_squares_y)

    #No_of_discretization must be even
    #num_squares_x = int((num_squares_x - (num_squares_x % 2))/2)
    #num_squares_y = int((num_squares_y - (num_squares_y % 2))/2)

    #x_step = (side - (num_squares * (cell_length))/num_squares
    x_step = (side/num_squares_x) - cell_length	
    y_step = (side/num_squares_y) - cell_length
    #print(num_squares, space_left)

    #Total_number_of_cells = num_squares_x * num_squares_y
    #print("Total_number_of_cells", Total_number_of_cells)
    #print("num_squares_x", num_squares_x, "num_squares_y", num_squares_y)
    #print("x_step", x_step, "y_step", y_step)

    #print('')
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

    #Special case. 0 will wreak havoc on the code.
    #The origin can never be (0,0) or (side, side)
    if origin[0] == 0 or origin[0] == side:
        for i in range(10):
            origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
            if origin[0] != 0.0 and origin[0] != side: break
    
    #Special case
    if origin[1] == 0 or origin[1] == side:
        for i in range(10):
            origin = np.random.uniform(low=0, high=side, size=(1, 3))[0]
            if origin[1] != 0.0 and origin[1] != side: break

    origin[2] = 0.0

    #print("origin", origin)

    #assert step >= 2*buffer, "no_of_discretization "+str(step)+" is too large for the cut_off" + " " + str(buffer)
    assert x_step >= 2, "Size of cell is too large "+str(x_step)+" is too large for the cut_off"
    assert y_step >= 2, "Size of cell is too large "+str(y_step)+" is too large for the cut_off"

    #The system is divided into 4 sets of cells  
    A_set = []
    #B_set = []
    #C_set = []
    #D_set = []

    x_pos = origin[0]
    y_pos = origin[1]
    
    rest_of_the_molecules_mask = np.ones(len(system_array_xyz), dtype=bool)

    #print("no_of_discretization", no_of_discretization)

    for i in range(num_squares_x):
        
        #x limits of the cell
        x_lower = x_pos
        x_upper = x_pos+cell_length

        #x limits of the cell with buffer
        x_lower_border = x_lower - buffer
        x_upper_border = x_upper + buffer

        #Where the middle of the cell is - i.e if the cell is split
        #If cell is not split, x lower split = x upper split
        #If cell is split, x lower split = 0 and x upper split = side
        x_lower_split = (x_lower + x_upper )/2
        x_upper_split = (x_lower + x_upper )/2

        if x_lower_border < 0: x_lower_border += side
        
        if x_lower < 0:
            x_lower += side
            x_lower_split = 0
            x_upper_split = side

        if x_upper_border > side: x_upper_border -= side

        if x_upper > side:
            x_upper -= side
            x_lower_split = 0
            x_upper_split = side

        #Note for unsplit cell x lower < x upper
        #For split cell x upper < x lower as the x upper is now on the other half of the cell due to PBC
        assert x_upper_split == side or x_upper_split == x_lower_split
        assert x_lower_split == 0 or x_lower_split == x_upper_split

        assert 0 < x_lower_border < side
        assert 0 < x_upper_border < side

        y_pos = origin[1]

        for j in range(num_squares_y):
                        
            y_lower = y_pos
            y_upper = y_pos+cell_length
            
            y_lower_border = y_lower - buffer
            y_upper_border = y_upper + buffer

            y_lower_split = (y_lower + y_upper )/2
            y_upper_split = (y_lower + y_upper )/2

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

            assert y_upper_split == side or y_upper_split == y_lower_split
            assert y_lower_split == 0 or y_lower_split == y_upper_split

            assert 0 < y_lower_border < side
            assert 0 < y_upper_border < side

            #If the cell is not split, the condition is simple
            if x_upper_split == x_lower_split:
                #Get all molecules whose x and y coordinates are within the cell
                cond_main_x =  (x_lower < system_array_xyz[:,0]) & (system_array_xyz[:,0] < x_upper )
            #The cell is split in x direction
            else:
                #Get all molecules between 0 and x upper
                cond_main_x_left =  ((x_upper >= system_array_xyz[:,0]) & (system_array_xyz[:,0] > x_lower_split)) 
                #Get all molecules between x lower and side
                cond_main_x_right =  ((x_upper_split > system_array_xyz[:,0]) & (system_array_xyz[:,0] >= x_lower ))
                #Merge both conditions
                cond_main_x = np.logical_or(cond_main_x_left, cond_main_x_right)

            if y_upper_split == y_lower_split:
                cond_main_y =  (y_lower < system_array_xyz[:,1]) & (system_array_xyz[:,1] < y_upper )
            else:
                cond_main_y_left =  ((y_upper >= system_array_xyz[:,1]) & (system_array_xyz[:,1] > y_lower_split))
                cond_main_y_right = ((y_upper_split > system_array_xyz[:,1]) & (system_array_xyz[:,1] >= y_lower ))
                cond_main_y = np.logical_or(cond_main_y_left, cond_main_y_right)

            #Merge the x and y conditions
            cond_main =  cond_main_x & cond_main_y
            
            #These are the molecules in the main box
            mol_main_box = system_array_xyz[cond_main]
            rest_of_the_molecules_mask[cond_main] = False

            #Now we will get the molecules in the border (buffer zone)
            #The border (buffer zone) is divided into 8 parts
            #Bottom left corner, bottom, bottom right corner, right,
            #top right corner, top, top left corner, left

            ####################
            #Bottom

            #The lower border is normal and the cell is unsplit
            if x_upper_split == x_lower_split:
                mask_b_x =  (x_lower <= system_array_xyz[:,0]) & (system_array_xyz[:,0] <= x_upper )
            #The lower border is split. The cell is split
            elif x_upper_split == side and x_lower_split == 0:
                #molecules between 0 and x upper
                mask_b_x_left =  ((system_array_xyz[:,0] <= x_upper ) & (x_lower_split < system_array_xyz[:,0])) 
                #molecules between x lower and side
                mask_b_x_right =  ((system_array_xyz[:,0] < x_upper_split ) & (x_lower <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_b_x = np.logical_or(mask_b_x_left, mask_b_x_right) 


            #The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower :
                #Molecules between y lower border and y lower
                mask_b_y = (y_lower_border < system_array_xyz[:,1] ) & (system_array_xyz[:,1] <= y_lower )
            #The lower border is split. The cell is split
            elif y_lower < y_lower_border :
                #molecules between 0 and y lower
                mask_b_y_bottom =  ((system_array_xyz[:,1] <= y_lower ) & (0 < system_array_xyz[:,1])) 
                #molecules between y lower border and side
                mask_b_y_top =  ((system_array_xyz[:,1] < side ) & (y_lower_border <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_b_y = np.logical_or(mask_b_y_bottom, mask_b_y_top)            
            else: raise Exception("Something is wrong")

            #Combine x and y conditiosn
            mask_b = mask_b_x & mask_b_y
            #Get the molecules in the bottom border
            mol_b = system_array_xyz[mask_b]
            #rest_of_the_molecules_mask[mask_b] = False

            ####################
            #Bottom left corner

            #The right border is normal. The cell is unsplit
            if x_lower_border < x_lower :
                #Molecules between x upper and y upper border
                mask_blc_x = (x_lower_border < system_array_xyz[:,0] ) & (system_array_xyz[:,0] <= x_lower )
            #The right border is split. The cell is split
            elif x_lower < x_lower_border :
                mask_blc_x_left =  ((system_array_xyz[:,0] <= x_lower ) & (0 < system_array_xyz[:,0])) 
                mask_blc_x_right =  ((system_array_xyz[:,0] < side ) & (x_lower_border <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_blc_x = np.logical_or(mask_blc_x_left, mask_blc_x_right)
            else: raise Exception("Something is wrong")

            #The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower :
                #Molecules between y lower border and y lower
                mask_blc_y = (y_lower_border < system_array_xyz[:,1] ) & (system_array_xyz[:,1] <= y_lower )
            #The lower border is split. The cell is split
            elif y_lower < y_lower_border :
                #molecules between 0 and y lower
                mask_blc_y_bottom =  ((system_array_xyz[:,1] <= y_lower ) & (0 < system_array_xyz[:,1])) 
                #molecules between y lower border and side
                mask_blc_y_top =  ((system_array_xyz[:,1] < side ) & (y_lower_border <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_blc_y = np.logical_or(mask_blc_y_bottom, mask_blc_y_top)            
            else: raise Exception("Something is wrong")

            #Molecules in the bottom left corner
            mask_blc = mask_blc_x & mask_blc_y
            mol_blc = system_array_xyz[mask_blc]
            #rest_of_the_molecules_mask[mask_blc] = False
           
            ####################
            #Bottom right corner
            #The small square of bottom left corner is normal and is in middle of the box
            if x_upper_border > x_upper :
                mask_brc_x = (x_upper <= system_array_xyz[:,0] ) & (system_array_xyz[:,0] < x_upper_border)
            #The small square of bottom left corner is split and has teleported to left side due to PBC
            elif x_upper > x_upper_border: 
                #mask_brc_x = (0 < system_array_xyz[:,0] ) & (system_array_xyz[:,0] <= x upper_border)
                mask_brc_x_left =  ((system_array_xyz[:,0] <= x_upper_border ) & (0 < system_array_xyz[:,0])) 
                mask_brc_x_right =  ((system_array_xyz[:,0] < side ) & (x_upper <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_brc_x = np.logical_or(mask_brc_x_left, mask_brc_x_right)
            else:
                raise Exception("Something is wrong")

            #The lower border is normal. The cell is unsplit
            if y_lower_border < y_lower :
                #Molecules between y lower border and y lower
                mask_brc_y = (y_lower_border < system_array_xyz[:,1] ) & (system_array_xyz[:,1] <= y_lower )
            #The lower border is split. The cell is split
            elif y_lower < y_lower_border :
                #molecules between 0 and y lower
                mask_brc_y_bottom =  ((system_array_xyz[:,1] <= y_lower ) & (0 < system_array_xyz[:,1])) 
                #molecules between y lower border and side
                mask_brc_y_top =  ((system_array_xyz[:,1] < side ) & (y_lower_border <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_brc_y = np.logical_or(mask_brc_y_bottom, mask_brc_y_top)            
            else: raise Exception("Something is wrong")

            mask_brc = mask_brc_x & mask_brc_y
            mol_brc = system_array_xyz[mask_brc]
            #rest_of_the_molecules_mask[mask_brc] = False

            ####################
            #Right

            #The right border is normal. The cell is unsplit
            if x_upper < x_upper_border:
                #Molecules between x upper and y upper border
                mask_r_x = (x_upper <= system_array_xyz[:,0] ) & (system_array_xyz[:,0] < x_upper_border)
            #The right border is split. The cell is split
            elif x_upper_border < x_upper :
                mask_r_x_left =  ((system_array_xyz[:,0] <= x_upper_border ) & (0 < system_array_xyz[:,0])) 
                mask_r_x_right =  ((system_array_xyz[:,0] < side ) & (x_upper <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_r_x = np.logical_or(mask_r_x_left, mask_r_x_right)
            else: raise Exception("Something is wrong")

            #The right border is normal. The cell is unsplit
            if y_upper_split == y_lower_split:
                mask_r_y =  (y_lower < system_array_xyz[:,1]) & (system_array_xyz[:,1] < y_upper )
            #The right border is split. The cell is split
            elif y_upper_split == side and y_lower_split == 0:
                mask_r_y_left =  ((y_upper >= system_array_xyz[:,1]) & (system_array_xyz[:,1] > y_lower_split))
                mask_r_y_right = ((y_upper_split > system_array_xyz[:,1]) & (system_array_xyz[:,1] >= y_lower ))
                mask_r_y = np.logical_or(mask_r_y_right, mask_r_y_left)

            mask_r = mask_r_x & mask_r_y
            mol_r = system_array_xyz[mask_r]
            #rest_of_the_molecules_mask[mask_r] = False

            ####################
            #Top right corner

            #The right border is normal. The cell is unsplit
            if x_upper < x_upper_border:
                #Molecules between x upper and y upper border
                mask_trc_x = (x_upper <= system_array_xyz[:,0] ) & (system_array_xyz[:,0] < x_upper_border)
            #The right border is split. The cell is split
            elif x_upper_border < x_upper :
                mask_trc_x_left =  ((system_array_xyz[:,0] <= x_upper_border ) & (0 < system_array_xyz[:,0])) 
                mask_trc_x_right =  ((system_array_xyz[:,0] < side ) & (x_upper <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_trc_x = np.logical_or(mask_trc_x_left, mask_trc_x_right)
            else: raise Exception("Something is wrong")
    
            #The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_trc_y = (y_upper <= system_array_xyz[:,1] ) & (system_array_xyz[:,1] < y_upper_border)
            #The lower border is split. The cell is split
            elif y_upper_border < y_upper :
                mask_trc_y_bottom =  ((system_array_xyz[:,1] <= y_upper_border ) & (0 < system_array_xyz[:,1])) 
                mask_trc_y_top =  ((system_array_xyz[:,1] < side ) & (y_upper <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_trc_y = np.logical_or(mask_trc_y_bottom, mask_trc_y_top)            
            else: raise Exception("Something is wrong")
            
            mask_trc = mask_trc_x & mask_trc_y
            mol_trc = system_array_xyz[mask_trc]
            #rest_of_the_molecules_mask[mask_trc] = False
            
            ####################
            #Top

            #The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_t_y = (y_upper <= system_array_xyz[:,1] ) & (system_array_xyz[:,1] < y_upper_border)
            #The lower border is split. The cell is split
            elif y_upper_border < y_upper :
                mask_t_y_bottom =  ((system_array_xyz[:,1] <= y_upper_border ) & (0 < system_array_xyz[:,1])) 
                mask_t_y_top =  ((system_array_xyz[:,1] < side ) & (y_upper <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_t_y = np.logical_or(mask_t_y_bottom, mask_t_y_top)            
            else: raise Exception("Something is wrong")
            
            if x_upper_split == x_lower_split:
                mask_t_x =  (x_lower < system_array_xyz[:,0]) & (system_array_xyz[:,0] < x_upper )
            elif x_upper_split == side and x_lower_split == 0:
                mask_t_x_left =  ((x_upper >= system_array_xyz[:,0]) & (system_array_xyz[:,0] > x_lower_split)) 
                mask_t_x_right =  ((x_upper_split > system_array_xyz[:,0]) & (system_array_xyz[:,0] >= x_lower ))
                mask_t_x = np.logical_or(mask_t_x_left, mask_t_x_right)
            
            mask_t = mask_t_x & mask_t_y
            mol_t = system_array_xyz[mask_t]
            #rest_of_the_molecules_mask[mask_t] = False

            ####################
            #Top left corner

            #The right border is normal. The cell is unsplit
            if x_lower_border < x_lower :
                #Molecules between x upper and y upper border
                mask_tlc_x = (x_lower_border < system_array_xyz[:,0] ) & (system_array_xyz[:,0] <= x_lower )
            #The right border is split. The cell is split
            elif x_lower < x_lower_border :
                mask_tlc_x_left =  ((system_array_xyz[:,0] <= x_lower ) & (0 < system_array_xyz[:,0])) 
                mask_tlc_x_right =  ((system_array_xyz[:,0] < side ) & (x_lower_border <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_tlc_x = np.logical_or(mask_tlc_x_left, mask_tlc_x_right)
            else: raise Exception("Something is wrong")
    
            #The lower border is normal. The cell is unsplit
            if y_upper < y_upper_border:
                mask_tlc_y = (y_upper <= system_array_xyz[:,1] ) & (system_array_xyz[:,1] < y_upper_border)
            #The lower border is split. The cell is split
            elif y_upper_border < y_upper :
                mask_tlc_y_bottom =  ((system_array_xyz[:,1] <= y_upper_border ) & (0 < system_array_xyz[:,1])) 
                mask_tlc_y_top =  ((system_array_xyz[:,1] < side ) & (y_upper <= system_array_xyz[:,1]))
                #Combine both conditions
                mask_tlc_y = np.logical_or(mask_tlc_y_bottom, mask_tlc_y_top)            
            else: raise Exception("Something is wrong")
            
            mask_tlc = mask_tlc_x & mask_tlc_y
            mol_tlc = system_array_xyz[mask_tlc]
            #rest_of_the_molecules_mask[mask_tlc] = False
            
            #####################
            #Left

            #The right border is normal. The cell is unsplit
            if x_lower_border < x_lower :
                #Molecules between x upper and y upper border
                mask_l_x = (x_lower_border < system_array_xyz[:,0] ) & (system_array_xyz[:,0] <= x_lower )
            #The right border is split. The cell is split
            elif x_lower < x_lower_border :
                mask_l_x_left =  ((system_array_xyz[:,0] <= x_lower ) & (0 < system_array_xyz[:,0])) 
                mask_l_x_right =  ((system_array_xyz[:,0] < side ) & (x_lower_border <= system_array_xyz[:,0]))
                #Combine both conditions
                mask_l_x = np.logical_or(mask_l_x_left, mask_l_x_right)
            else: raise Exception("Something is wrong")
            
            if y_upper_split == y_lower_split:
                mask_l_y =  (y_lower < system_array_xyz[:,1]) & (system_array_xyz[:,1] < y_upper )
            elif y_upper_split == side and y_lower_split == 0:
                mask_l_y_left =  ((y_upper >= system_array_xyz[:,1]) & (system_array_xyz[:,1] > y_lower_split))
                mask_l_y_right = ((y_upper_split > system_array_xyz[:,1]) & (system_array_xyz[:,1] >= y_lower ))
                mask_l_y = np.logical_or(mask_l_y_left, mask_l_y_right)
            
            mask_l = mask_l_x & mask_l_y
            mol_l = system_array_xyz[mask_l]
            #rest_of_the_molecules_mask[mask_l] = False

            ####################

            main_mol = mol_main_box
            border_mol = np.vstack((mol_blc, mol_b, mol_brc, mol_r, mol_trc, mol_t, mol_tlc, mol_l))

            #plt.scatter(mol_b[:,0], mol_b[:,1], c='b', s=100, alpha=0.15)
            #plt.scatter(mol_t[:,0], mol_t[:,1], c='r', s=100, alpha=0.15)

            cell_dims = [[x_lower, x_upper,
                         y_lower, y_upper],
                         [x_lower_split, x_upper_split,
                         y_lower_split, y_upper_split],
                         [x_lower_border, x_upper_border,
                         y_lower_border, y_upper_border]]
            
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
            
            if y_pos > side: y_pos -= side
                    
        x_pos += x_step + cell_length
        if x_pos > side: x_pos -= side

    #plt.show()

    rest_of_the_molecules_xyz = system_array_xyz[rest_of_the_molecules_mask]

    '''
    import matplotlib.pyplot as plt

    #plt.scatter(rest_of_the_molecules_xyz[:,0], rest_of_the_molecules_xyz[:,1], s=1, alpha=0.15, color = 'black')

    counter = 0
    for mol_xyz, border_mol, cell_dims in A_set:
        
        counter += len(mol_xyz) 
        #counter += len(border_mol)

        plt.scatter(mol_xyz[:,0], mol_xyz[:,1], c='r', s=1, alpha=0.15)
        
        plt.scatter(border_mol[:,0], border_mol[:,1], c='b', s=1, alpha=0.15)

    counter += len(rest_of_the_molecules_xyz)


    assert counter == len(system_array_xyz), str(counter) + " " + str(len(system_array_xyz))

    plt.show()
    '''

    '''
    fig, ax = plt.subplots(dpi=300)
    #plt.figure(dpi=300, alpha=0.15)
    #plt.axis("equal")
    #plt.figure.set_aspect('equal', 'box')
    ax.set_aspect('equal', 'box')

    #ax.scatter(system_array_xyz[:,0], system_array_xyz[:,1], s=1, alpha=0.15, color = 'black')

    for mol_xyz, border_mol, cell_dims in A_set:
        
        ax.scatter(mol_xyz[:,0], mol_xyz[:,1], c='r', s=1)
        ax.scatter(border_mol[:,0], border_mol[:,1], c='b', s=1)

    ax.axis("off")
    fig.savefig(str(x_pos)+"_"+str(y_pos)+".png")
    print(str(x_pos)+"_"+str(y_pos)+" saved! \n")
    plt.show()
    '''
    
    return A_set, rest_of_the_molecules_xyz	

if __name__ == "__main__":
    
    side = 200
    Total_area = side**2
    radius_of_disk = 0.5
    area_of_disk = np.pi * radius_of_disk * radius_of_disk
    num_mol = int(5.1 * (Total_area/area_of_disk))
    #######################################
    molecules_xyz = np.random.uniform(low=0, high=side, size=(num_mol, 3))
    molecules_xyz[:,2] = 0

    for i in range(1):
        i += 0.1
        #grid_up_the_system(side=side, no_of_discretization=8, buffer=5,
        #                   system_array_xyz=molecules_xyz, x_pos = i, y_pos = i)
        grid_up_the_system(system_array_xyz=molecules_xyz)
    
    print("Done")
