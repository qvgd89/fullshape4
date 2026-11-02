import os
import numpy as np
import lsstypes as types
from lsstypes import ObservableLeaf, ObservableTree
from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles
from lsstypes import Mesh3SpectrumPole, Mesh3SpectrumPoles
from lsstypes import read

pole_selection=[True,True,True,True,True]
range1=[0.00,100]
ranges=[range1,range1,range1,range1,range1]

def build_pk_data_synthetic(tracer, z_string, model = 'model1', k_min_p = 0.02, k_max_p0 = 0.2, k_max_p2 = 0.2, k_max_b0 = 0.2, k_max_b2 = 0.03, Vol = 1):
    data = ExtractSyntheticData(tracer, z_string, model, k_min_p, k_max_p0, k_max_p2, k_max_b0, k_max_b2)
    covariances = ExtractDataEZmock(tracer, z_string, k_min_p, k_max_p0, k_max_p2, k_max_b0, k_max_b2, Vol)
    
    return data | covariances

def ExtractSyntheticData(tracer, z_string, model = 'model1', k_min_p = 0.02, k_max_p0 = 0.2, k_max_p2 = 0.2, k_max_b0 = 0.2, k_max_b2 = 0.03):

    base_path = '/global/cfs/cdirs/desi/science/cpe/elbers/neutrino_mock_challenge/synthetic_data/'
    
    synthetic_data_table = np.loadtxt(base_path + "synthetic_vector_property_split_" + model + "_" + z_string + ".txt")

    k_edges = np.vstack([np.linspace(0.0025, 0.3975, 80), np.linspace(0.0075, 0.4025, 80)]).T
    k_data = synthetic_data_table[:,0]
    pkl0 = synthetic_data_table[:,1]
    pkl2 = synthetic_data_table[:,2]
    pkl4 = synthetic_data_table[:,3]    
    
    # Build Mesh Spectrum object
    synthetic_shot_noise = 1700.0
    pk_0 = Mesh2SpectrumPole(k=k_data, k_edges=k_edges, num_raw=pkl0, num_shotnoise = np.ones_like(pkl0) * synthetic_shot_noise)
    pk_2 = Mesh2SpectrumPole(k=k_data, k_edges=k_edges, num_raw=pkl2)
    pk_4 = Mesh2SpectrumPole(k=k_data, k_edges=k_edges, num_raw=pkl4)
    
    pk_0 = pk_0.select(k=(k_min_p, k_max_p0))
    pk_2 = pk_2.select(k=(k_min_p, k_max_p2))
    pk_4 = pk_4.select(k=(k_min_p, k_max_p2))
    
    spectrum2 = Mesh2SpectrumPoles([pk_0, pk_2], ells=[0, 2])
    
    return ({'k_data': k_data, 'pk_lsstypes': spectrum2, 'pkl0': pk_0, 'pkl2': pk_2, 'pkl4': pk_4})

def ExtractDataEZmock(tracer, z_string, k_min_p = 0.02, k_max_p0 = 0.2, k_max_p2 = 0.2, k_max_b0 = 0.2, k_max_b2 = 0.03, Vol = 1.):
    
    if (tracer == 'LRG1' or tracer == 'LRG2'):
        tracer = 'LRG'

    mocks='EZmock'
    
    print("Cov kmax_0", k_max_b0)
    print("Cov kmax_2", k_max_b2)
    
    base_path = '/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/sugiyama_basis/SecondGenMocks/'
    
    if (mocks=='EZmock'): folder_box='/CubicBox_6Gpc/'
    if (mocks=='AbacusSummit'): folder_box='/CubicBox/'
    
    path_bisp=base_path+mocks+folder_box+tracer+'/'+z_string+'/diag/bispec/'
    path_pk=base_path+mocks+folder_box+tracer+'/'+z_string+'/diag/powspec/'

    nummocks=2000
    
    observables = []
    
    k_edges = np.vstack([np.linspace(0.0025, 0.3975, 80), np.linspace(0.0075, 0.4025, 80)]).T
    k_edges_Bk1 = np.asarray([np.linspace(0.0025, 0.3975, 80), np.linspace(0.0025, 0.3975, 80)])
    k_edges_Bk2 = np.asarray([np.linspace(0.0075, 0.4025, 80), np.linspace(0.0075, 0.4025, 80)])
    k_edges_Bk = np.asarray([k_edges_Bk1, k_edges_Bk2]).T
    
    print(k_min_p, k_max_p0, k_max_p2, k_max_b0, k_max_b2)
    
    # Check if this covariance matrix has already been constructed
    covmat_path = 'observable_covmats/' + tracer + '_' + z_string + '_' + str(nummocks) + '_' + str(k_min_p) + '_' + str(k_max_p0) + '_' + str(k_max_p2) + '_' + str(k_max_b0) + '_' + str(k_max_b2) + '/covariance.h5'
    covmat_exist = os.path.isfile(covmat_path)
    
    if not covmat_exist:
        for i in range(nummocks):
            j=i+1
            if (0 <= j < 10): seed='000'+str(j)
            if (10 <= j < 100): seed='00'+str(j)
            if (100 <= j < 1000): seed='0'+str(j)
            if (1000 <= j < 10000): seed=str(j)
            fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
            fileB202 = np.loadtxt(path_bisp+'bk202_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
            filepkl0 = np.loadtxt(path_pk  +'pk0_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
            filepkl2 = np.loadtxt(path_pk  +'pk2_'       +tracer+'_'+z_string+'_seed'+seed, unpack = True)
            
            if (j==1):
                k_eff_all = fileB000[1]
                k_eff_all_Bk = np.vstack([k_eff_all, k_eff_all]).T
            
            noisel0 = filepkl0[5]
            noisel2 = filepkl2[5]
            noiseB000 = fileB000[8]
            noiseB202 = fileB202[8]
            
            pk_0 = Mesh2SpectrumPole(k=k_eff_all, k_edges=k_edges, num_raw=filepkl0[3], num_shotnoise = noisel0)
            pk_2 = Mesh2SpectrumPole(k=k_eff_all, k_edges=k_edges, num_raw=filepkl2[3], num_shotnoise = noisel2)
            Bk_000 = Mesh3SpectrumPole(k=k_eff_all_Bk, k_edges=k_edges_Bk, num_raw=fileB000[6], num_shotnoise = noiseB000, ell=(0,0,0), basis='sugiyama-diagonal')
            Bk_202 = Mesh3SpectrumPole(k=k_eff_all_Bk, k_edges=k_edges_Bk, num_raw=fileB202[6], num_shotnoise = noiseB202, ell=(2,0,2), basis='sugiyama-diagonal')
            
            pk_0 = pk_0.select(k=(k_min_p, k_max_p0))
            pk_2 = pk_2.select(k=(k_min_p, k_max_p2))
            if not k_max_b0 == None:
                Bk_000 = Bk_000.select(k=(k_min_p, k_max_b0))
            if not k_max_b2 == None:
                Bk_202 = Bk_202.select(k=(k_min_p, k_max_b2))
            
            spectrum2 = Mesh2SpectrumPoles([pk_0, pk_2], ells=[0, 2])
            spectrum3 = Mesh3SpectrumPoles([Bk_000, Bk_202], ells=[(0,0,0), (2,0,2)])
            
            # spectrum2.write("/global/cfs/cdirs/desi/science/cpe/elbers/summary_statistics/box/ezmock/mesh2_spectrum_poles_LRG_z%.3f_c000_los-z_%03d.h5" % (float(z_string[1:]), i))
            # spectrum3.write("/global/cfs/cdirs/desi/science/cpe/elbers/summary_statistics/box/ezmock/mesh3_spectrum_poles_LRG_z%.3f_c000_los-z_%03d.h5" % (float(z_string[1:]), i))
            
            tree = ObservableTree(
                [spectrum2, spectrum3],
                observables=['spectrum2', 'spectrum3']
            )
            observables.append(tree)
        
        covariance = types.cov(observables)
        
        print(f"  Storing covariance at", covmat_path)
        covariance.write(covmat_path)
    else:
        print(f"  Reading covariance from", covmat_path)
        covariance = read(covmat_path)
        
        # Reading the k vector
        seed = '0001'
        fileB000 = np.loadtxt(path_bisp+'bk000_diag_'+tracer+'_'+z_string+'_seed'+seed, unpack = True)
        k_eff_all = fileB000[1]
        
    print(f"  Rescaling covariance by {Vol} to account for volume")
    covariance._value *= Vol
        
    cov = covariance.value()
    print(f"  Full Cov shape: {cov.shape}")
    cov_bk = covariance.at.observable.get(observables='spectrum3')
    cov_pk = covariance.at.observable.get(observables='spectrum2')

    return ({'k': k_eff_all, 'cov_array': cov, 'cov_pk': cov_pk, 'cov_bk': cov_bk}) 
