import numpy as np
def split_patches(x, y, z, colsplit, rowsplit, choose_patches):
    """
    A program for randomly choosing patches of a given size from a 
    two dimensional data set

    param: x: array of x points, typically a linspace between 0 and 1. 
    Must match z in x-dimension
    param: y: array of y points, typically a linspace between 0 and 1. 
    Must match z in y-dimension
    param: z: two-dimensional array of datapoints
    param: colsplit: How many column(c) do you want the data split into
    param: rowsplit: How many rows(r) do you want the data split into
    param: choose_patches: Number of patches your want to keep. Can not be bigger than [c x r]
    type: x: np.ndarray
    type: y: np.ndarray
    type: z: np.ndarray
    type: colsplit: int
    type: rowsplit: int
    type: choose_patches: int
    return: x(1darray), y(1darray), z(1darray)
    """

    x,y = np.meshgrid(x,y)


    ############
    #Spliting The Data
    ############
    zcolsplit = np.array_split(z, colsplit, axis=1)
    xcolsplit = np.array_split(x, colsplit, axis=1)
    ycolsplit = np.array_split(y, colsplit, axis=1)
    zsplit = [np.array_split(np.vstack(zcolsplit[i]), rowsplit) for i in range(len(zcolsplit))]
    xsplit = [np.array_split(np.vstack(xcolsplit[i]), rowsplit) for i in range(len(xcolsplit))]
    ysplit = [np.array_split(np.vstack(ycolsplit[i]), rowsplit) for i in range(len(ycolsplit))]

    ###Choosing random indices
    ## and making sure no set of 
    ###indices are equal
    np.random.choice(rowsplit, choose_patches)
    randomset =  []
    while len(frozenset(randomset)) < choose_patches:
        a = np.random.choice(colsplit)
        b = np.random.choice(rowsplit)
        randomset.append((a,b))
    

    z = np.concatenate([zsplit[randset[0]][randset[1]].reshape(-1,1) for randset in randomset])
    x = np.concatenate([xsplit[randset[0]][randset[1]].reshape(-1,1) for randset in randomset])
    y = np.concatenate([ysplit[randset[0]][randset[1]].reshape(-1,1) for randset in randomset])
    return x, y, z
