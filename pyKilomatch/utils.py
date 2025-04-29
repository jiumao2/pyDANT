import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csgraph

def computeKernel2D(xp, yp, sig=20):
    '''Compute the 2D kernel matrix for the given points xp and yp.

    Arguments:
        - xp: 2D array of points (n_samples, 2)
        - yp: 2D array of points (n_samples, 2)
        - sig: standard deviation for the Gaussian kernel (default: 20)

    Returns:
        - K: kernel matrix (n_samples_xp, n_samples_yp)
    '''

    distx = np.abs(
        np.expand_dims(xp[:,0], 1) - np.expand_dims(yp[:,0], 0))
    disty = np.abs(
        np.expand_dims(xp[:,1], 1) - np.expand_dims(yp[:,1], 0))

    sigx = sig
    sigy = 1.5 * sig

    p = 1
    K = np.exp(-(distx / sigx)**p - (disty / sigy)**p)
    return K

def graphEditNumber(matA, matB):
    ''' Compute the merge number of two graphs A and B and the number of same merges.

    Arguments:
        - matA: connectivity matrix of graph A (n_nodes_A, n_nodes_A)
        - matB: connectivity matrix of graph B (n_nodes_B, n_nodes_B)

    Returns:
        - nSame: number of same merges
        - nA: number of merges in graph A
        - nB: number of merges in graph B
    '''

    # Convert to sparse matrices and find connected components
    comp_A = csgraph.connected_components(matA, directed=False)[1]
    comp_B = csgraph.connected_components(matB, directed=False)[1]
    comp_AB = csgraph.connected_components(matA*matB, directed=False)[1]
    
    # Count edges within components (n-1 edges per n-node component)
    nA = sum([np.sum(comp_A == i)-1 for i in np.unique(comp_A)])
    nB = sum([np.sum(comp_B == i)-1 for i in np.unique(comp_B)])
    nSame = sum([np.sum(comp_AB == i)-1 for i in np.unique(comp_AB)])
    return (nSame, nA, nB)

def spikeLocation(waveforms_mean, channel_locations, n_nearest_channels=20, algorithm='monopolar_triangulation'):
    '''Spike location estimation using either center_of_mass or monopolar_triangulation
    
    monopolar_triangulation: refer to Boussard, Julien, Erdem Varol, Hyun Dong Lee, Nishchal Dethe, and Liam Paninski. “Three-Dimensional Spike Localization and Improved Motion Correction for Neuropixels Recordings.” In Advances in Neural Information Processing Systems, 34:22095–105. Curran Associates, Inc., 2021. https://proceedings.neurips.cc/paper/2021/hash/b950ea26ca12daae142bd74dba4427c8-Abstract.html.
    > https://spikeinterface.readthedocs.io/en/stable/modules/postprocessing.html#spike-locations
    > https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/postprocessing/localization_tools.py#L334

    Arguments:
        - waveforms_mean: mean waveforms (n_channels, n_samples)
        - channel_locations: 2D array of channel locations (n_channels, 2)
        - n_nearest_channels: number of nearest channels to consider for localization, default is 20
        - algorithm: 'center_of_mass' or 'monopolar_triangulation', default is 'monopolar_triangulation'

    returns:
        - x: x coordinate of the spike location
        - y: y coordinate of the spike location
        - z: z coordinate of the spike location
        - ptt: peak-to-trough value of the spike waveform

    '''

    # get n_nearest_channels from the channels with the largest peak-to-trough value
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
        [x0[0] - 100, x0[1] - 100, 1, 0],
        [x0[0] + 100, x0[1] + 100, 100 * 10, 1000*ptt_max],
    )

    output = least_squares(fun, x0=x0, bounds=bounds, args=(ptt_this, loc_this))
    
    return tuple(output["x"])

def waveformEstimation(waveform_mean, location, channel_locations, location_new):
    '''Waveform estimation with Kriging interpolation.

    Arguments:
        - waveform_mean: mean waveform (n_channels, n_samples)
        - location: original location of the spike (x, y), 1D array of length 2
        - channel_locations: 2D array of channel locations (n_channels, 2)
        - location_new: new location of the spike (x, y), 1D array of length 2
    
    Returns:
        - waveform_out: estimated waveform at the new location (n_samples)
    '''
    
    # Calculate mapped location
    location_mapped_to_old = channel_locations - (np.array(location_new) - np.array(location))
    
    # 2D coordinates for interpolation
    xp = channel_locations
    
    # 2D kernel of the original channel positions
    Kxx = computeKernel2D(xp, xp)
    
    # 2D kernel of the new channel positions
    yp = location_mapped_to_old
    Kyx = computeKernel2D(yp, xp)
    
    # Kernel prediction matrix
    M = Kyx @ np.linalg.inv(Kxx + 0.01 * np.eye(Kxx.shape[0]))
    
    # Interpolated waveform at the new location
    waveform_out = M @ waveform_mean
    
    return waveform_out

