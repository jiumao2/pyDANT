import numpy as np

def computeKernel2D(xp, yp, sig=None):
    if sig is None:
        sig = 20

    distx = np.abs(
        np.expand_dims(xp[:,0], 1) - np.expand_dims(yp[:,0], 0))
    disty = np.abs(
        np.expand_dims(xp[:,1], 1) - np.expand_dims(yp[:,1], 0))

    sigx = sig
    sigy = 1.5 * sig

    p = 1
    K = np.exp(-(distx / sigx)**p - (disty / sigy)**p)
    return K

def findNearestPoints(points, queryPoints):
    pass

def graphEditNumber(matA, matB):
    from scipy.sparse import csgraph
    
    # Convert to sparse matrices and find connected components
    comp_A = csgraph.connected_components(matA, directed=False)[1]
    comp_B = csgraph.connected_components(matB, directed=False)[1]
    comp_AB = csgraph.connected_components(matA*matB, directed=False)[1]
    
    # Count edges within components (n-1 edges per n-node component)
    nA = sum([np.sum(comp_A == i)-1 for i in np.unique(comp_A)])
    nB = sum([np.sum(comp_B == i)-1 for i in np.unique(comp_B)])
    nSame = sum([np.sum(comp_AB == i)-1 for i in np.unique(comp_AB)])
    return (nSame, nA, nB)

def spikeLocation(waveforms_mean, chanMap, n_nearest_channels=None, algorithm=None):
    from scipy.optimize import least_squares
    
    if n_nearest_channels is None:
        n_nearest_channels = 20
    if algorithm is None:
        algorithm = 'monopolar_triangulation'

    # get n_nearest_channels from the channels with the largest peak-to-trough value
    channel_locations = np.column_stack((chanMap["xcoords"], chanMap["ycoords"]))

    peaks_to_trough = np.max(waveforms_mean, axis=1) - np.min(waveforms_mean, axis=1)
    idx_max = np.argmax(peaks_to_trough)

    loc_max = channel_locations[idx_max, :]
    distance_to_max = np.sum((channel_locations - loc_max)**2, axis=1)

    idx_sorted = np.argsort(distance_to_max)
    idx_included = idx_sorted[:n_nearest_channels]

    # calculate the center_to_mass location
    ptt_max = peaks_to_trough[idx_max]
    ptt_this = peaks_to_trough[idx_included]
    loc_this = channel_locations[idx_included,:]

    loc_center_to_mass = np.sum(loc_this * ptt_this[:, np.newaxis], axis=0) / np.sum(ptt_this)

    if algorithm.lower() == 'center_of_mass':
        x = loc_center_to_mass[0]
        y = loc_center_to_mass[1]
        z = 0
        ptt = ptt_max

        return x, y, z, ptt

    # calculate the monopolar_triangulation location
    def fun(x, ptt, loc_this):
        ptt_estimated = x[3] / np.sqrt((loc_this[:,0]-x[0])**2 + (loc_this[:,1]-x[1])**2 + x[2]**2)
        return ptt - ptt_estimated

    x0 = [loc_center_to_mass[0], loc_center_to_mass[1], 1, ptt_max]
    bounds = (
        [x0[0] - 1000, x0[1] - 1000, 1, 0],
        [x0[0] + 1000, x0[1] + 1000, 1000 * 10, 1000*ptt_max],
    )

    output = least_squares(fun, x0=x0, bounds=bounds, args=(ptt_this, loc_this))
    
    return tuple(output["x"])

def waveformEstimation(waveform_mean, location, chanMap, location_new, x, y):
    # Filter connected channels
    channel_locations = np.column_stack((chanMap["xcoords"], chanMap["ycoords"]))
    
    # Calculate mapped location
    location_mapped_to_old = np.array([x, y]) - (np.array(location_new) - np.array(location))
    
    n_channels = 36
    distance_to_location = np.sum((channel_locations - np.array([x, y]))**2, axis=1)
    
    idx_sorted = np.argsort(distance_to_location)
    idx_included = idx_sorted[:n_channels]
    
    # 2D coordinates for interpolation
    xp = channel_locations[idx_included, :]
    
    # 2D kernel of the original channel positions
    Kxx = computeKernel2D(xp, xp)
    
    # 2D kernel of the new channel positions
    yp = location_mapped_to_old[:2]  # Use only x,y coordinates
    Kyx = computeKernel2D(yp[np.newaxis, :], xp)
    
    # Kernel prediction matrix
    M = Kyx @ np.linalg.inv(Kxx + 0.01 * np.eye(Kxx.shape[0]))
    
    waveform_out = np.sum(waveform_mean[idx_included, :] * M.T, axis=0)
    
    return waveform_out



def waveformSimilarity(waveforms, waveform_channels, n_channels=None):
    if n_channels is None:
        n_channels = 36

    # similarityAB
    channels = waveform_channels[0, :n_channels]
    idx_channels_A = np.arange(n_channels)
    idx_channels_B = np.zeros(n_channels, dtype=int)
    
    for k in range(n_channels):
        temp = np.where(waveform_channels[1,:] == channels[k])[0]
        if len(temp) == 0:
            return 0
            
        idx_channels_B[k] = temp[0]
    
    waveformsA = waveforms[0, idx_channels_A, :].reshape(1, -1)
    waveformsB = waveforms[1, idx_channels_B, :].reshape(1, -1)
    
    temp = np.corrcoef(waveformsA, waveformsB)
    similarityAB = np.atanh(temp[0,1])

    # similarityBA
    channels = waveform_channels[1, :n_channels]
    idx_channels_A = np.zeros(n_channels, dtype=int)
    idx_channels_B = np.arange(n_channels)
    
    for k in range(n_channels):
        temp = np.where(waveform_channels[0,:] == channels[k])[0]
        if len(temp) == 0:
            return 0
            
        idx_channels_A[k] = temp[0]
    
    waveformsA = waveforms[0, idx_channels_A, :].reshape(1, -1)
    waveformsB = waveforms[1, idx_channels_B, :].reshape(1, -1)
    
    temp = np.corrcoef(waveformsA, waveformsB)
    similarityBA = np.atanh(temp[0,1])

    return np.max([similarityAB, similarityBA])

def computeSimilarity(x, y):
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0

    temp = np.corrcoef(x,y)
    similarity = np.atanh(temp[0,1])
    if np.isnan(similarity):
        similarity = 0

    return similarity


