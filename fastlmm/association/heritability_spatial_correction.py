from __future__ import absolute_import
from __future__ import print_function
from pysnptools.util.mapreduce1.runner import *
import logging
import fastlmm.pyplink.plink as plink
import pysnptools.util.pheno as pstpheno
import pysnptools.util as pstutil
import numpy as np
import scipy.stats as stats
from pysnptools.snpreader import Bed
from pysnptools.standardizer import DiagKtoN
from pysnptools.kernelreader import KernelData, KernelNpz
import time
import pandas as pd
from fastlmm.inference.lmm_cov import LMM as fastLMM
from fastlmm.inference.lmm import LMM
import sklearn.metrics
from pysnptools.snpreader import Pheno
from pysnptools.standardizer import Unit
import itertools
from sklearn import model_selection
from scipy import stats
from six.moves import map
from six.moves import range


def _nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

def _write_csv(dataframe, index, filename):
    if os.path.exists(filename):
        os.remove(filename)
    dataframe.to_csv(filename,sep="\t",index=index)
    logging.info(filename)
    logging.info(dataframe)

def spatial_similarity(spatial_coor, alpha, power):     # scale spatial coordinates
    """
    :param spatial_coor: The position of each individual given by two coordinates. Any units are allowed, but the two values
       must be compatible so that distance can be determined via Pythagoras' theorem. (So, longitude and latitude should
       not be used unless the locations are near the Equator.) 
    :type spatial_coor: a iid_count x 2 array

    :param alpha: a similarity scale. The similarity of two individuals is defined as exp(-(distance_between/alpha)**power).
    :type alpha: number

    :param power: 2 (a good choice) means that similarity goes with area. 1 means with distance.
    :type alpha: number

    :rtype: square numpy array of similarities.
    """
    return np.exp(-
                  (sklearn.metrics.pairwise_distances(spatial_coor) /alpha)**power
                  )

def work_item(arg_tuple):
    return work_item2(*arg_tuple)

def work_item2(pheno, G_kernel, spatial_coor, spatial_iid, alpha, alpha_power, xxx_todo_changeme, xxx_todo_changeme1, xxx_todo_changeme2,
     just_testing, do_uncorr, do_gxe2, a2): 
    
    #########################################
    # Load GPS info from filename if that's the way it is given
    ########################################
    (jackknife_index, jackknife_count, jackknife_seed) = xxx_todo_changeme
    (permute_plus_index, permute_plus_count, permute_plus_seed) = xxx_todo_changeme1
    (permute_times_index, permute_times_count, permute_times_seed) = xxx_todo_changeme2
    if isinstance(spatial_coor,str):
        assert spatial_iid is None, "if spatial_coor is a str, then spatial_iid should be None"
        gps_table = pd.read_csv(spatial_coor, delimiter=" ").dropna()
        spatial_iid = np.array([(v,v) for v in gps_table["id"].values])
        spatial_coor = gps_table[["south_new", "east_new"]].values


    #########################################
    # Remove any missing values from pheno
    ########################################
    assert pheno.sid_count == 1, "Expect only one pheno in work_item"
    pheno = pheno.read()
    pheno = pheno[pheno.val[:,0]==pheno.val[:,0],:] #Excludes NaN because NaN is not equal to NaN

    #########################################
    # Environment: Turn spatial info info a KernelData
    #########################################
    spatial_val = spatial_similarity(spatial_coor, alpha, power=alpha_power)
    E_kernel = KernelData(iid=spatial_iid,val=spatial_val)

    #########################################
    # Intersect, apply the jackknife or permutation, and then (because we now know the iids) standardize appropriately
    #########################################
    from pysnptools.util import intersect_apply
    G_kernel, E_kernel, pheno  = intersect_apply([G_kernel, E_kernel, pheno])

    if jackknife_index >= 0:
        assert jackknife_count <= G_kernel.iid_count, "expect the number of groups to be less than the number of iids"
        assert jackknife_index < jackknife_count, "expect the jackknife index to be less than the count"
        m_fold = model_selection.KFold(n_splits=jackknife_count, shuffle=True, random_state=jackknife_seed%4294967295).split(list(range(G_kernel.iid_count)))
        iid_index,_ = _nth(m_fold, jackknife_index)
        pheno = pheno[iid_index,:]
        G_kernel = G_kernel[iid_index]
        E_kernel = E_kernel[iid_index]

    if permute_plus_index >= 0:
        #We shuffle the val, but not the iid, because that would cancel out.
        #Integrate the permute_plus_index into the random.
        np.random.seed((permute_plus_seed + permute_plus_index)%4294967295)
        new_index = np.arange(G_kernel.iid_count)
        np.random.shuffle(new_index)
        E_kernel_temp = E_kernel[new_index].read()
        E_kernel = KernelData(iid=E_kernel.iid,val=E_kernel_temp.val,name="permutation {0}".format(permute_plus_index))

    pheno = pheno.read().standardize()       # defaults to Unit standardize
    G_kernel = G_kernel.read().standardize() # defaults to DiagKtoN standardize
    E_kernel = E_kernel.read().standardize() # defaults to DiagKtoN standardize

    #########################################
    # find h2uncoor, the best mixing weight of pure random noise and G_kernel
    #########################################

    if not do_uncorr:
        h2uncorr, nLLuncorr = np.nan,np.nan
    else:
        logging.info("Find best h2 for G_kernel")
        lmmg = LMM()
        lmmg.setK(K0=G_kernel.val)
        lmmg.setX(np.ones([G_kernel.iid_count,1])) # just a bias column
        lmmg.sety(pheno.val[:,0])
        if not just_testing:
            resg = lmmg.findH2()
            h2uncorr, nLLuncorr = resg["h2"], resg["nLL"]
        else:
            h2uncorr, nLLuncorr = 0,0
        logging.info("just G: h2uncorr: {0}, nLLuncorr: {1}".format(h2uncorr,nLLuncorr))
    
    #########################################
    # Find a2, the best mixing for G_kernel and E_kernel
    #########################################

    if a2 is None:
        logging.info("Find best mixing for G_kernel and E_kernel")
        lmm1 = LMM()
        lmm1.setK(K0=G_kernel.val, K1=E_kernel.val, a2=0.5)
        lmm1.setX(np.ones([G_kernel.iid_count,1])) # just a bias column
        lmm1.sety(pheno.val[:,0])
        if not just_testing:
            res1 = lmm1.findA2()
            h2, a2, nLLcorr = res1["h2"], res1["a2"], res1["nLL"]
            h2corr = h2 * (1-a2)
            e2 = h2 * a2
            h2corr_raw = h2
        else:
            h2corr, e2, a2, nLLcorr, h2corr_raw = 0,0,.5,0,0
        logging.info("G plus E mixture: h2corr: {0}, e2: {1}, a2: {2}, nLLcorr: {3} (h2corr_raw:{4})".format(h2corr,e2,a2,nLLcorr,h2corr_raw))
    else:
        h2corr, e2, nLLcorr, h2corr_raw = np.nan, np.nan, np.nan, np.nan

    #########################################
    # Find a2_gxe2, the best mixing for G+E_kernel and the GxE kernel
    #########################################

    if not do_gxe2:
        gxe2, a2_gxe2, nLL_gxe2 = np.nan, np.nan, np.nan
    else:
        #Create the G+E kernel by mixing according to a2
        val=(1-a2)*G_kernel.val + a2*E_kernel.val
        GplusE_kernel = KernelData(iid=G_kernel.iid, val=val,name="{0} G + {1} E".format(1-a2,a2))
        #Don't need to standardize GplusE_kernel because it's the weighted combination of standardized kernels

        # Create GxE Kernel and then find the best mixing of it and GplusE
        logging.info("Find best mixing for GxE and GplusE_kernel")

        val=G_kernel.val * E_kernel.val
        if permute_times_index >= 0:
            #We shuffle the val, but not the iid, because doing both would cancel out
            np.random.seed((permute_times_seed + permute_times_index)%4294967295)
            new_index = np.arange(G_kernel.iid_count)
            np.random.shuffle(new_index)
            val = pstutil.sub_matrix(val, new_index, new_index)

        GxE_kernel = KernelData(iid=G_kernel.iid, val=val,name="GxE") # recall that Python '*' is just element-wise multiplication
        GxE_kernel = GxE_kernel.standardize()

        lmm2 = LMM()
        lmm2.setK(K0=GplusE_kernel.val, K1=GxE_kernel.val, a2=0.5)
        lmm2.setX(np.ones([G_kernel.iid_count,1])) # just a bias column
        lmm2.sety(pheno.val[:,0])
        if not just_testing:
            res2 = lmm2.findA2()
            gxe2, a2_gxe2, nLL_gxe2 = res2["h2"], res2["a2"], res2["nLL"]
            gxe2 *= a2_gxe2
        else:
            gxe2, a2_gxe2, nLL_gxe2 = 0,.5,0
        logging.info("G+E plus GxE mixture: gxe2: {0}, a2_gxe2: {1}, nLL_gxe2: {2}".format(gxe2, a2_gxe2, nLL_gxe2))
        
    #########################################
    # Return results
    #########################################

    ret = {"h2uncorr": h2uncorr, "nLLuncorr": nLLuncorr, "h2corr": h2corr, "h2corr_raw": h2corr_raw,"e2":e2, "a2": a2, "nLLcorr": nLLcorr,
           "gxe2": gxe2, "a2_gxe2": a2_gxe2, "nLL_gxe2": nLL_gxe2, "alpha": alpha, "alpha_power":alpha_power, "phen": np.array(pheno.sid,dtype='str')[0],
           "jackknife_index": jackknife_index, "jackknife_count":jackknife_count, "jackknife_seed":jackknife_seed,
           "permute_plus_index": permute_plus_index, "permute_plus_count":permute_plus_count, "permute_plus_seed":permute_plus_seed,
           "permute_times_index": permute_times_index, "permute_times_count":permute_times_count, "permute_times_seed":permute_times_seed
           }
    
    logging.info("run_line: {0}".format(ret))
    return ret

def heritability_spatial_correction(G_kernel, spatial_coor, spatial_iid, alpha_list, alpha_power, pheno, 
                     map_function = map, cache_folder=None, 
                     jackknife_count=500, permute_plus_count=10000, permute_times_count=10000, seed=0,
                     just_testing=False,  always_remote=False, allow_gxe2 = True,
                     count_A1=None
                     ):
    """
    Function measuring heritability with correction for spatial location.

    :param G_kernel: A kernel that tells the genetic similarity between all pairs of individuals. The kernel can be given 
      explicitly, for example with a :class:`.KernelData`. The kernel can also be given implicitly by providing a set of
      SNPs or the name of a BED file.
    :type G_kernel: a `KernelReader <http://fastlmm.github.io/PySnpTools/#kernelreader-kernelreader>`__, `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param spatial_coor: The position of each individual given by two coordinates. Any units are allowed, but the two values
       must be compatible so that distance can be determined via Pythagoras' theorem. (So, longitude and latitude should
       not be used unless the locations are near the Equator.) 
    :type spatial_coor: a iid_count x 2 array

    :param spatial_iid: A ndarray of the iids. Each iid is a ndarray of two strings (a family ID and a case ID) that identifies an individual.
    :type spatial_iid: array of strings with shape [iid_count,2]

    :param alpha_list: a list of numbers to search to find the best alpha, which is the similarity scale. The similarity of two individuals
      is here defined as exp(-(distance_between/alpha)**alpha_power). If the closest individuals are 100 units apart and the farthest
      individuals are 4e6 units apart, a reasonable alpha_list might be: [int(v) for v in np.logspace(np.log10(100),np.log10(1e10), 100)]
      The function's reports on the alphas chosen. If an extreme alpha is picked, change alpha_list to cover more range.
    :type alpha_list: list of numbers

    :param alpha_power: 2 (a good choice) means that similarity goes with area. 1 means with distance.
    :type alpha_list: number

    :param pheno: The target values(s) to predict. It can be a file name readable via `Pheno <http://fastlmm.github.io/PySnpTools/#snpreader-pheno>`__ or any `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__.
    :type pheno: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or string

    :param cache_folder: (default 'None') The name of a directory in which to save intermediate results. If 'None', then no intermediate results are saved.
    :type cache_folder: a string

    :param map_function: (default 'map') A function with the same inputs and functionality as Python's 'map' function.
       Can be used to run 'heritability_spatial_correction' on a cluster.
    :type map_function: a function

    :param jackknife_count: (default 500) The number of jackknife groups to use when calculating standard errors (SE). Changing to a small number, 2, 
       speeds up calculation at the cost of unusable SEs.
    :type jackknife_count: number

    :param permute_plus_count: (default 10000) The number of permutations used when calculating P values. Changing to a small number, 1, 
       speeds up calculation at the cost of unusable P values.
    :type permute_plus_count: number

    :param permute_times_count: (default 10000) The number of permutations used when calculating P values. Changing to a small number, 1, 
       speeds up calculation at the cost of unusable P values.
    :type permute_times_count: number

    :param seed: (default 0) The random seed used by jackknifing and permutation.
    :type seed: number

    :param just_testing: (default False) If true, skips actual LMM-related search and calculation.
    :type just_testing: bool

    :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
         alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
    :type count_A1: bool

    :rtype: Pandas dataframe with one row per phenotype. Columns include "h2uncorr", "h2corr", etc.

    """

    ######################
    # Prepare the inputs
    ######################

    from fastlmm.inference.fastlmm_predictor import _kernel_fixup, _pheno_fixup
    G_kernel = _kernel_fixup(G_kernel, iid_if_none=None, standardizer=Unit(),count_A1=count_A1)  # Create a kernel from an in-memory kernel, some snps, or a text file.
    pheno = _pheno_fixup(pheno,iid_if_none=G_kernel.iid, missing='NA',count_A1=count_A1) # Create phenotype data from in-memory data or a text file.

    if cache_folder is not None:
        pstutil.create_directory_if_necessary(cache_folder,isfile=False)

    
    jackknife_seed = seed or 1954692566
    permute_plus_seed = seed or 2372373100
    permute_times_seed = seed or 2574440128

    ######################
    # Find 'alpha', the scale for distance
    ######################

    # create the alpha table (unless it is already there)
    alpha_table_fn = "{0}/alpha_table.{1}.txt".format(cache_folder,pheno.sid_count) # create a name for the alpha_table cache file
    phen_target_array = np.array(pheno.sid,dtype='str')
    if cache_folder is not None and os.path.exists(alpha_table_fn):
        alpha_table = pd.read_csv(alpha_table_fn, delimiter = '\t',index_col=False, comment=None)
    else:
        # create the list of arguments to run    
        arg_list = []   
        for phen_target in phen_target_array:
            pheno_one = pheno[:,pheno.col_to_index([phen_target])] # Look at only this pheno_target
            for alpha in alpha_list:
                            #pheno, G_kernel, spatial_coor, spatial_iid, alpha,     alpha_power,  (jackknife_index, jackknife_count, jackknife_seed),
                arg_tuple = (pheno_one, G_kernel, spatial_coor, spatial_iid, alpha, alpha_power, (-1,     0,     None),  
                             # (permute_plus_index, permute_plus_count, permute_plus_seed), (permute_times_index, permute_times_count, permute_times_seed) ,just_testing, do_uncorr, do_gxe2,               a2
                               (-1,     0,     None),                                       (-1,     0,     None),                                          just_testing, False,     True and allow_gxe2,   None)
                arg_list.append(arg_tuple)

        # Run "run_line" on each set of arguments and save to file
        return_list = map_function(work_item, arg_list) if len(arg_list)>1 or always_remote else list(map(work_item, arg_list))
        return_list = [line for line in return_list if line is not None] #Remove 'None' results
        alpha_table = pd.DataFrame(return_list)
        if cache_folder is not None:
            _write_csv(alpha_table,False,alpha_table_fn)

    # read the alpha table and find the best values
    grouped = alpha_table.groupby("phen")
    alpha_dict = {}
    for phen, phen_table in grouped:
        best_index_corr = phen_table['nLLcorr'].idxmin() # with Pandas, this returns the index in the parent table, not the group table
        best_index_gxe2 = phen_table['nLL_gxe2'].idxmin() if allow_gxe2 else 0
        alpha_corr = alpha_table.iloc[best_index_corr]['alpha']
        alpha_gxe2 = alpha_table.iloc[best_index_gxe2]['alpha']
        alpha_dict[phen] = alpha_corr, alpha_gxe2
    logging.info(alpha_dict)


    ######################
    # Use jackknifing to compute h2uncorr, SE, h2corr, SE, e2, SE, gxe2, SE
    ######################

    jackknife_count_actual = min(jackknife_count,G_kernel.iid_count)

    # Set up the run and do it (unless it has already been run)
    jackknife_table_fn = "{0}/jackknife.{1}.count{2}.txt".format(cache_folder, pheno.sid_count, jackknife_count_actual)
    if cache_folder is not None and os.path.exists(jackknife_table_fn):
        jackknife_table = pd.read_csv(jackknife_table_fn, delimiter = '\t',index_col=False, comment=None)
    else:
        arg_list = []
        for phen_target in phen_target_array:
            pheno_one = pheno[:,pheno.col_to_index([phen_target])] # Look at only this pheno_target
            alpha_corr, alpha_gxe2 = alpha_dict[phen_target]
            alpha_set = set([alpha_corr, alpha_gxe2]) #If these are the same, then only need to do half the work
            for alpha in alpha_set:
                logging.debug(alpha)
                do_uncorr = (alpha == alpha_corr)
                do_gxe2   = (alpha == alpha_gxe2) and allow_gxe2
                for jackknife in range(-1, jackknife_count_actual):
                               # pheno, G_kernel, spatial_coor, spatial_iid, alpha,     alpha_power, (jackknife_index, jackknife_count,         jackknife_seed),
                    arg_tuple = (pheno_one, G_kernel, spatial_coor, spatial_iid, alpha, alpha_power, (jackknife,       jackknife_count_actual,  jackknife_seed),
                                    # (permute_plus_index, permute_plus_count, permute_plus_seed), (permute_times_index, permute_times_count, permute_times_seed) ,just_testing, do_uncorr, do_gxe2, a2
                                    (-1,0,None),                                                 (-1,0,None),                                                    just_testing, do_uncorr, do_gxe2, None)
                    arg_list.append(arg_tuple)    

        # Run "run_line" on each set of arguments and save to file
        return_list = map_function(work_item, arg_list) if len(arg_list)>1 or always_remote else list(map(work_item, arg_list))
        return_list = [line for line in return_list if line is not None] #Remove 'None' results
        jackknife_table = pd.DataFrame(return_list)
        if cache_folder is not None:
            _write_csv(jackknife_table, False, jackknife_table_fn)


    # get the real (that is, unjackknifed) values    
    jackknife_table["diff"] = jackknife_table.h2uncorr-jackknife_table.h2corr # Compute the diff = h2uncorr-h2corr column
    results_both = jackknife_table[jackknife_table.jackknife_index==-1]  # Create a table of the real (non-jackknifed) results for both alphas (which may be the same)
    del results_both["jackknife_index"]
    results_corr = results_both[results_both.alpha == [alpha_dict[phen][0] for phen in results_both.phen]] #Create version for g+e's alpha
    results_gxe2 = results_both[results_both.alpha == [alpha_dict[phen][1] for phen in results_both.phen]] #Create version for gxe's alpha
    #remove unwanted columns
    for delcol in ["a2_gxe2","gxe2","nLL_gxe2","permute_plus_count","permute_plus_index","permute_plus_seed","permute_times_count","permute_times_index","permute_times_seed","jackknife_count","jackknife_seed"]:
        del results_corr[delcol]
    for delcol in ["a2","e2","h2corr","h2uncorr","nLLcorr","nLLuncorr","diff","permute_plus_count","permute_plus_index","permute_plus_seed","permute_times_count","permute_times_index","permute_times_seed","jackknife_count","jackknife_seed"]:
        del results_gxe2[delcol]

    if jackknife_count_actual > 0:
        #Use a pivottable to compute the jackknifed SE's
        corr_rows = np.logical_and(jackknife_table.jackknife_index!=-1,jackknife_table.alpha==[alpha_dict[phen][0] for phen in jackknife_table.phen])
        jk_table_corr = pd.pivot_table(jackknife_table[corr_rows], values=['h2uncorr','h2corr','diff','e2'], index=['phen'], columns=[], aggfunc=np.std)
        jk_table_corr["h2uncorr SE"] = jk_table_corr["h2uncorr"] * np.sqrt(jackknife_count_actual-1)
        jk_table_corr["h2corr SE"] = jk_table_corr["h2corr"] * np.sqrt(jackknife_count_actual-1)
        jk_table_corr["diff SE"] = jk_table_corr["diff"] * np.sqrt(jackknife_count_actual-1)
        jk_table_corr["e2 SE"] = jk_table_corr["e2"] * np.sqrt(jackknife_count_actual-1)
        del jk_table_corr["h2uncorr"]
        del jk_table_corr["h2corr"]
        del jk_table_corr["diff"]
        del jk_table_corr["e2"]
        gxe2_rows = np.logical_and(jackknife_table.jackknife_index!=-1,jackknife_table.alpha==[alpha_dict[phen][1] for phen in jackknife_table.phen])
        jk_table_gxe2 = pd.pivot_table(jackknife_table[gxe2_rows], values=['gxe2'], index=['phen'], columns=[], aggfunc=np.std)
        jk_table_gxe2["gxe2 SE"] = jk_table_gxe2["gxe2"] * np.sqrt(jackknife_count_actual-1)
        del jk_table_gxe2["gxe2"]

        #Join the SE's to the main results table
        results_corr = results_corr.join(jk_table_corr, on='phen')
        results_gxe2 = results_gxe2.join(jk_table_gxe2, on='phen')
    else:
        for col in ['h2uncorr SE','h2corr SE','diff SE','e2 SE']:
            results_corr[col] = np.NaN
        results_gxe2['gxe2 SE'] = np.NaN

    #compute pValue columns
    results_corr["P (diff=0)"] = stats.t.sf(results_corr["diff"]/results_corr["diff SE"],df=jackknife_count_actual-1)*2 #two sided
    results_corr["from SE, one-sided, P (e2=0)"] = stats.t.sf(results_corr["e2"]/results_corr["e2 SE"],df=jackknife_count_actual-1)
    results_gxe2["from SE, one-sided, P (gxe2=0)"] = stats.t.sf(results_gxe2["gxe2"]/results_gxe2["gxe2 SE"],df=jackknife_count_actual-1)   #one sided

    if cache_folder is not None:
        _write_csv(results_corr, False, "{0}/jackknife_corr_summary.{1}.jackknife{2}.txt".format(cache_folder, pheno.sid_count, jackknife_count_actual))
        _write_csv(results_gxe2, False, "{0}/jackknife_gxe2_summary.{1}.jackknife{2}.txt".format(cache_folder, pheno.sid_count, jackknife_count_actual))


    ######################
    # compute p(e2=0) via permutation
    ######################

    permplus_table_fn = "{0}/permutation.GPlusE.{1}.count{2}.txt".format(cache_folder, pheno.sid_count, permute_plus_count)
    if cache_folder is not None and os.path.exists(permplus_table_fn):
        permplus_table = pd.read_csv(permplus_table_fn, delimiter = '\t',index_col=False, comment=None)
    else:
        arg_list = []
        for phen_target in phen_target_array:
            pheno_one = pheno[:,pheno.col_to_index([phen_target])] # Look at only this pheno_target
            alpha_corr, alpha_gxe2 = alpha_dict[phen_target]
            for jackknife_index in range(-1,permute_plus_count):
                           # pheno, G_kernel, spatial_coor, spatial_iid, alpha,          alpha_power,    (jackknife_index, jackknife_count, jackknife_seed),
                arg_tuple = (pheno_one, G_kernel, spatial_coor, spatial_iid, alpha_corr, alpha_power, (-1,0,None),
                             # (permute_plus_index, permute_plus_count, permute_plus_seed), (permute_times_index, permute_times_count, permute_times_seed) ,just_testing, do_uncorr, do_gxe2, a2
                             (jackknife_index, permute_plus_count,permute_plus_seed),       (-1,0,None),                                                    just_testing, False,    False,    None)
                arg_list.append(arg_tuple)

        # Run "run_line" on each set of arguments and save to file
        return_list = map_function(work_item, arg_list) if len(arg_list)>1 or always_remote else list(map(work_item, arg_list))
        return_list = [line for line in return_list if line is not None] #Remove 'None' results
        permplus_table = pd.DataFrame(return_list)
        if cache_folder is not None:
            _write_csv(permplus_table, False, permplus_table_fn)


    #Create a table of the real nLL for each pheno
    real_result_permplus = permplus_table[permplus_table.permute_plus_index==-1][['phen','nLLcorr']]
    real_result_permplus.rename(columns={'nLLcorr':'nLLcorr_real'},inplace=True)
    real_result_permplus.set_index(['phen'],inplace=True)

    # Create a table of the permutation runs and add the real nLL to each row
    perm_table = permplus_table[permplus_table.permute_plus_index!=-1]
    result = perm_table.join(real_result_permplus, on='phen')
    result['P(e2)'] = [1.0 if b else 0.0 for b in result.nLLcorr <= result.nLLcorr_real] # create a column showing where the perm is better (or as good) as the real
    # Use pivottable to find the fraction of of times when permutation is better
    pivot_table_plus = pd.pivot_table(result, values=['P(e2)'], index=['phen'], columns=[], aggfunc=np.mean)
    if cache_folder is not None:
        summary_permplus_table_fn = "{0}/summary.permutation.GPlusE.{1}.count{2}.txt".format(cache_folder, pheno.sid_count, permute_plus_count)
        _write_csv(pivot_table_plus, True, summary_permplus_table_fn)

    ################################################
    # compute p(gxe2=0) via permutation
    ################################################

    #Only process phenos for which gxe2 is not 0
    nonzero = set(results_gxe2[results_gxe2.gxe2 !=0].phen)
    permtimes_phenotypes = set(phen_target_array) & nonzero #intersection
    permtimes_table_list = []
    for phen_target in permtimes_phenotypes:
        permtimes_table_fn = "{0}/permutation.GxE/{1}.count{2}.txt".format(cache_folder, phen_target, permute_times_count)

        if cache_folder is not None and os.path.exists(permtimes_table_fn):
            permtime_results = pd.read_csv(permtimes_table_fn, delimiter = '\t',index_col=False, comment=None)
        else:
            arg_list = []
            pheno_one = pheno[:,pheno.col_to_index([phen_target])] # Look at only this pheno_target
            alpha_corr, alpha_gxe2 = alpha_dict[phen_target]
            a2 = float(permplus_table[permplus_table.phen==phen_target][permplus_table.permute_plus_index == -1]['a2'])
            for permute_index in range(-1,permute_times_count):
                           # pheno, G_kernel, spatial_coor, spatial_iid, alpha,          alpha_powerm (permute_index, permute_count, permute_seed),
                arg_tuple = (pheno_one, G_kernel, spatial_coor, spatial_iid, alpha_gxe2, alpha_power, (-1,0,None),
                             # (permute_plus_index, permute_plus_count, permute_plus_seed), (permute_times_index, permute_times_count, permute_times_seed) ,just_testing, do_uncorr, do_gxe2, a2
                            (-1,0,None),                                                    (permute_index, permute_times_count,permute_times_seed),        just_testing, False,     allow_gxe2,    a2)
                arg_list.append(arg_tuple)    

            # Run "run_line" on each set of arguments and save to file
            return_list = map_function(work_item, arg_list) if len(arg_list)>1 or always_remote else list(map(work_item, arg_list))
            return_list = [line for line in return_list if line is not None] #Remove 'None' results
            permtime_results = pd.DataFrame(return_list)
            if cache_folder is not None:
                pstutil.create_directory_if_necessary(permtimes_table_fn)
                _write_csv(permtime_results,False,permtimes_table_fn)
        permtimes_table_list.append(permtime_results)

    if permtimes_table_list: #not empty
        permtimes_table = pd.concat(permtimes_table_list)
        logging.info(permtimes_table.head())

        #Create a table of the real nLL for each pheno
        real_result_permtimes = permtimes_table[permtimes_table.permute_times_index==-1][['phen','nLL_gxe2']]
        real_result_permtimes.rename(columns={'nLL_gxe2':'nLL_gxe2_real'},inplace=True)
        real_result_permtimes.set_index(['phen'],inplace=True)

        # Create a table of the permutation runs and add the real nLL to reach row
        summary_permtimes_table_fn = "{0}/summary.permutation.GxE.{1}.count{2}.txt".format(cache_folder,len(permtimes_phenotypes), permute_times_count)

        perm_table = permtimes_table[permtimes_table.permute_times_index!=-1]
        resultx = perm_table.join(real_result_permtimes, on='phen')
        resultx['P(gxe2)'] = [1.0 if b else 0.0 for b in resultx.nLL_gxe2 <= resultx.nLL_gxe2_real] # create a column showing where the perm is better (or as good) as the real
        # Use pivottable to find the fraction of times when permutation is better
        pivot_table_times = pd.pivot_table(resultx, values=['P(gxe2)'], index=['phen'], columns=[], aggfunc=np.mean)
        if cache_folder is not None:
            _write_csv(pivot_table_times,True,summary_permtimes_table_fn)


    #######################
    # Create final table of results by combining the summary tables
    #######################

    #Rename some columns
    results_corr.rename(columns={"h2uncorr SE":"SE (h2uncorr)","h2corr SE":"SE (h2corr)","e2 SE":"SE (e2)"}, inplace=True)

    #Rename some columns and join results
    results_gxe2.rename(columns={"alpha":"alpha_gxe2","gxe2 SE":"SE (gxe2)","h2corr_raw":"h2corr_raw_gxe2"}, inplace=True)
    del results_gxe2['alpha_power']
    results_gxe2.set_index(["phen"],inplace=True)
    final0 = results_corr.join(results_gxe2, on='phen')

    #Rename some columns and join results
    pivot_table_plus.rename(columns={"P(e2)":"P(e2=0)"}, inplace=True)
    if len(pivot_table_plus)>0:
        final1 = final0.join(pivot_table_plus, on='phen')
    else:
        final1 = final0.copy()
        final1['P(e2=0)'] = np.NaN

    #Rename some columns and join results
    if permtimes_table_list and len(pivot_table_times)>0: #not empty
        pivot_table_times.rename(columns={"P(gxe2)":"P(gxe2=0)"}, inplace=True)
        final2 = final1.join(pivot_table_times, on='phen')
    else:
        final2 = final1.copy()
        final2["P(gxe2=0)"] = np.nan

    #Rename 'phen' and select final columns
    final2.rename(columns={"phen":"phenotype"}, inplace=True)
    final3 = final2[["phenotype","h2uncorr","SE (h2uncorr)","h2corr","SE (h2corr)","P (diff=0)","e2","SE (e2)","P(e2=0)","alpha","alpha_gxe2","gxe2","SE (gxe2)","P(gxe2=0)"]].copy()

    #Rename sort the phenotypes
    final3['lower'] = [pheno_one.lower() for pheno_one in final3.phenotype]
    final3.sort_values(['lower'],inplace=True)
    del final3['lower']

    if cache_folder is not None:
        summary_final_table_fn = "{0}/summary.final.{1}.{2}.{3}.{4}.txt".format(cache_folder, pheno.sid_count, jackknife_count_actual,permute_plus_count,permute_times_count)
        _write_csv(final3,False,summary_final_table_fn)
    
    return final3

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    do_plot = False
    from pysnptools.util import snp_gen
    from pysnptools.standardizer import Unit

    seed = 0
    N = 5000

    #Generate SNPs
    snpdata = snp_gen(fst=.1, dfr=0, iid_count=N, sid_count=1000, chr_count=10, label_with_pop=True,seed=seed)
    K_causal = snpdata.read_kernel(Unit()).standardize()

    if do_plot:
        pylab.suptitle("$K_{causal}$")
        pylab.imshow(K_causal.val, cmap=pylab.gray(), vmin=0, vmax=1)
        pylab.show()

    import numpy as np
    from pysnptools.snpreader import SnpData

    distance_between_centers = 2500000
    x0 = distance_between_centers * 0.5
    x1 = distance_between_centers * 1.5
    y0 = distance_between_centers
    y1 = distance_between_centers
    sd = distance_between_centers/4.

    spatial_iid = snpdata.iid
    center_dict = {"0": (x0,y0), "1": (x1,y1)}
    centers = np.array([center_dict[iid_item[0]] for iid_item in spatial_iid])
    np.random.seed(seed)
    logging.info("Generating positions for seed {0}".format(seed))
    spatial_coor_gen_original = SnpData(iid=snpdata.iid,sid=["x","y"],val=centers+np.random.multivariate_normal([0,0],[[1,0],[0,1]],size=len(centers))*sd,parent_string="'spatial_coor_gen_original'")

    if do_plot:
        import matplotlib.pyplot as plt
        color_dict = {"0": "r", "1": "b", "2": "g"}
        colors = [color_dict[iid_item] for iid_item in snpdata.iid[:,0]]
        plt.axis('equal')
        plt.scatter(spatial_coor_gen_original.val[:,0], spatial_coor_gen_original.val[:,1], c=colors)
        plt.show()

    from fastlmm.association.heritability_spatial_correction import spatial_similarity
    from pysnptools.kernelreader import KernelData

    alpha = distance_between_centers
    spatial_val = spatial_similarity(spatial_coor_gen_original.val, alpha, power=2)
    K_loc = KernelData(iid=snpdata.iid,val=spatial_val).standardize()

    if do_plot:
        pylab.suptitle("$K_{loc}$")
        pylab.imshow(K_loc.val, cmap=pylab.gray(), vmin=0, vmax=1)
        pylab.show()

    from pysnptools.snpreader import SnpData

    iid = K_causal.iid
    iid_count = K_causal.iid_count
    np.random.seed(seed)
    pheno_causal = SnpData(iid=iid,sid=["causal"],val=np.random.multivariate_normal(np.zeros(iid_count),K_causal.val).reshape(-1,1),parent_string="causal")
    np.random.seed(seed^998372)
    pheno_noise = SnpData(iid=iid,sid=["noise"],val=np.random.normal(size=iid_count).reshape(-1,1),parent_string="noise")
    np.random.seed(seed^12230302)
    pheno_loc_original = SnpData(iid=iid,sid=["loc_original"],val=np.random.multivariate_normal(np.zeros(iid_count),K_loc.val).reshape(-1,1),parent_string="loc_original")

    do_shuffle = False
    if do_shuffle:
        idx = np.arange(iid_count)
        np.random.seed(seed)
        np.random.shuffle(idx)
        pheno_loc = pheno_loc_original.read(view_ok=True) #don't need to copy, because the next line will be fresh memory
        pheno_loc.val = pheno_loc.val[idx,:]
    else:
        pheno_loc = pheno_loc_original
                       
    pheno = SnpData(iid=iid,sid=["pheno_all"],val=pheno_causal.val+pheno_noise.val+pheno_loc.val)

    is_pop0 = snpdata.iid[:,0]=="0"
    is_pop1 = snpdata.iid[:,0]=="1"
    print("pop0 mean={0}. pop1 mean={1}".format(pheno.val[is_pop0].mean(),pheno.val[is_pop1].mean()))

    if do_plot:
        pylab.suptitle("Phenotype $y$")
        plt.plot(list(range(sum(is_pop0))),pheno.val[is_pop0],"r.",list(range(sum(is_pop1))),pheno.val[is_pop1],"b.")
        plt.show()

    from pysnptools.standardizer import Unit
    import numpy as np
    from fastlmm.association.heritability_spatial_correction import spatial_similarity
    from pysnptools.kernelreader import KernelData
    from pysnptools.snpreader import SnpData
    from fastlmm.association import heritability_spatial_correction

    def generate_and_analyze(seed, N, do_shuffle, just_testing=True, map_function=None, cache_folder=None):

        #Generate SNPs
        snpdata = snp_gen(fst=.1, dfr=0, iid_count=N, sid_count=1000, chr_count=10, label_with_pop=True,seed=seed)
        K_causal = snpdata.read_kernel(Unit()).standardize()

        #Generate geo-spatial locations and K_loc
        distance_between_centers = 2500000
        x0 = distance_between_centers * 0.5
        x1 = distance_between_centers * 1.5
        y0 = distance_between_centers
        y1 = distance_between_centers
        sd = distance_between_centers/4.

        spatial_iid = snpdata.iid
        center_dict = {"0": (x0,y0), "1": (x1,y1)}
        centers = np.array([center_dict[iid_item[0]] for iid_item in spatial_iid])
        np.random.seed(seed)
        logging.info("Generating positions for seed {0}".format(seed))
        spatial_coor = SnpData(iid=snpdata.iid,sid=["x","y"],val=centers+np.random.multivariate_normal([0,0],[[1,0],[0,1]],size=len(centers))*sd,parent_string="'spatial_coor_gen_original'")
        alpha = distance_between_centers
        spatial_val = spatial_similarity(spatial_coor.val, alpha, power=2)
        K_loc = KernelData(iid=snpdata.iid,val=spatial_val).standardize()

        #Generate phenotype
        iid = K_causal.iid
        iid_count = K_causal.iid_count
        np.random.seed(seed)
        pheno_causal = SnpData(iid=iid,sid=["causal"],val=np.random.multivariate_normal(np.zeros(iid_count),K_causal.val).reshape(-1,1),parent_string="causal")
        np.random.seed(seed^998372)
        pheno_noise = SnpData(iid=iid,sid=["noise"],val=np.random.normal(size=iid_count).reshape(-1,1),parent_string="noise")
        np.random.seed(seed^12230302)
        pheno_loc_original = SnpData(iid=iid,sid=["loc_original"],val=np.random.multivariate_normal(np.zeros(iid_count),K_loc.val).reshape(-1,1),parent_string="loc_original")

        if do_shuffle:
            idx = np.arange(iid_count)
            np.random.seed(seed)
            np.random.shuffle(idx)
            pheno_loc = pheno_loc_original.read(view_ok=True) #don't need to copy, because the next line will be fresh memory
            pheno_loc.val = pheno_loc.val[idx,:]
        else:
            pheno_loc = pheno_loc_original

        pheno = SnpData(iid=iid,sid=["pheno_all"],val=pheno_causal.val+pheno_noise.val+pheno_loc.val)

        #Analyze data
        alpha_list = [int(v) for v in np.logspace(np.log10(100),np.log10(1e10), 100)]
        dataframe = heritability_spatial_correction(
            snpdata,
            spatial_coor.val,spatial_iid,
            alpha_list=[alpha] if just_testing else alpha_list,pheno=pheno, alpha_power=2,
            jackknife_count=0,permute_plus_count=0,permute_times_count=0,just_testing=just_testing,
            map_function=map_function,cache_folder=cache_folder)

        logging.info(dataframe)
        return dataframe

    def loop_generate_and_analyze(N, seed_count, do_shuffle, just_testing, map_function, cache_folder=None):
        '''
        run generate_and_analyze on different seeds
        '''
        df_list = []
        for seed in range(seed_count):
            if cache_folder is not None:
                cache_folder_per_seed = cache_folder.format(seed)
            else:
                cache_folder_per_seed = None
            df = generate_and_analyze(seed=seed,N=N,do_shuffle=do_shuffle,
                               just_testing=just_testing, map_function=map_function, cache_folder=cache_folder_per_seed)
            df_list.append(df)
        
        #format the output dataframe            
        df2 = pd.concat(df_list)
        df2["seed"] = list(range(seed_count))
        df2["N"]=N
        df2["do_shuffle"]=do_shuffle
        df3 = df2[["seed","N","do_shuffle", "alpha","h2corr","e2","h2uncorr"]]
        df3.set_index(["seed"],inplace=True)
        return df3

    df = loop_generate_and_analyze(N=100,seed_count=7,do_shuffle=True,just_testing=True,map_function=map)
    df = loop_generate_and_analyze(N=50,seed_count=1,do_shuffle=False,just_testing=False,map_function=map)
    df = loop_generate_and_analyze(N=50,seed_count=1,do_shuffle=True,just_testing=False,map_function=map)
    df