#!/usr/bin/env python2.7
#
# Written (W) 2014 Christian Widmer
# Copyright (C) 2014 Microsoft Research

"""
Created on 2014-04-07
@author: Christian Widmer
@summary: Module containing helper function to evaluate quality of GWAS
"""

from __future__ import absolute_import
from __future__ import print_function
import time
import os.path

import numpy as np
import scipy as sp
from scipy import stats

from fastlmm.util.pickle_io import load
from fastlmm.util.util import excludeinds



def estimate_lambda(pv):
    """estimate lambda form a set of PV"""
    LOD2 = sp.median(stats.chi2.isf(pv,1))
    L = (LOD2/0.456)
    return (L)


def cut_snps_close_to_causals(p_values, pos, causal_idx, mindist, plot=False):

    i_causal_all = np.zeros(len(p_values), dtype=np.bool)
    i_causal_all[causal_idx] = True
        
    # cut out snps in LD to causal snps
    pos_all = np.array(pos)
    pos_causal = np.array(pos)[causal_idx]

    close_to_causals_mask = excludeinds(pos_all, pos_causal, mindist=mindist, idist=2)
    
    # keep causals and far away snps
    i_keepers = np.bitwise_or(i_causal_all, ~close_to_causals_mask)
        
    print(("keeping %i/%i SNPs (mindist=%f)" % (sum(i_keepers), len(i_keepers), mindist)))

    pv = p_values[i_keepers]
    #pv_lin = p_values_lin[i_keepers]
    i_causal = i_causal_all[i_keepers]

    if plot:
        import pylab
        pylab.plot(pos[i_keepers,2], pos[i_keepers,0], "+")
        pylab.plot(pos[:,2], pos[:,0]+0.5, "+")
        pylab.show()

    return pv, i_causal



def eval_gwas(pv, i_causal, out_fn=None, plot=False):
    """
    
    """

    pv_thresholds = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]

    # compute lambda on all p-values?
    #lambda_gc = estimate_lambda(p_values)
    lambda_gc = estimate_lambda(pv)
        
    n_causal = i_causal.sum()

    #compute power and Type-1 error
    power = np.zeros_like(pv_thresholds)
    t1err = np.zeros_like(pv_thresholds)
    power_corr = np.zeros_like(pv_thresholds)
    t1err_corr = np.zeros_like(pv_thresholds)
        
    pvcorr = stats.chi2.sf(stats.chi2.isf(pv,1)/lambda_gc,1)

    for i_t, t in enumerate(pv_thresholds):
        #compute uncorrected power and T1
        i_lower = pv<t
        power[i_t] =  i_causal[i_lower].sum()/(1.0*(n_causal))
        t1err[i_t] = (~i_causal[i_lower]).sum()/(1.0*(len(i_causal)-n_causal))

        #compute GC corrected Power and T1
        i_lower_corr = pvcorr<t
        power_corr[i_t] =  i_causal[i_lower_corr].sum()/(1.0*(n_causal))
        t1err_corr[i_t] = (~i_causal[i_lower_corr]).sum()/(1.0*(len(i_causal)-n_causal))


    if plot == True:
        import pylab
        pylab.figure()
        pylab.title("lambda_gc=%f" % lambda_gc)
        pylab.plot(pv_thresholds, power, "-o")
        pylab.yscale("log")
        pylab.xscale("log")
        pylab.xlabel("pv threshold")
        pylab.ylabel("power")
        pylab.grid(True)
        pylab.plot(pv_thresholds, power_corr, "-o")
            
        if not out_fn is None:
            pow_fn = out_fn.replace(".pickle", "_pow.pdf")
            pylab.savefig(pow_fn)
        else:
            pylab.show()

        pylab.figure()
        pylab.title("lambda_gc=%f" % lambda_gc)
        pylab.plot(pv_thresholds, t1err, "-o", label="t1err")
        pylab.plot(pv_thresholds, t1err_corr, "-o", label="t1err_gc")
        pylab.yscale("log")
        pylab.xscale("log")
        pylab.xlabel("pv threshold")
        pylab.ylabel("t1err")
        pylab.grid(True)
          
        pylab.plot(pv_thresholds, pv_thresholds, "-", label="thres")
        pylab.legend(loc="upper left")
        if not out_fn is None:
            t1err_fn = out_fn.replace(".pickle", "_t1err.pdf")
            pylab.savefig(t1err_fn)
        else:
            pylab.show()

        # plot auROC
        if out_fn is None:
            roc_fn = None
        else:
            roc_fn = out_fn.replace(".pickle", "_roc.pdf")
        plot_roc(i_causal, -pv, label='lambda_gc=%0.4f' % (lambda_gc), out_fn=roc_fn)

        # plot auPRC
        if out_fn is None:
            prc_fn = None
        else:
            prc_fn = out_fn.replace(".pickle", "_prc.pdf")
        plot_prc(i_causal, -pv, label='lambda_gc=%0.4f' % (lambda_gc), out_fn=prc_fn)


    # wrap up metrics
    res = {}
    res["lambda"] = lambda_gc
    res["pv_thresholds"] = pv_thresholds
    res["power"] = power
    res["power_corr"] = power_corr
    res["t1err"] = t1err
    res["t1err_corr"] = t1err_corr

    return res



###############################################################
# t1err
###############################################################

def plot_t1err_noshow(pv, i_causal, label="", gc_correct=False):
    """
    False Positive Rate = FP / N = 1 - specificity
    """
    pv_thresholds, t1err = compute_t1err_data(pv, i_causal)
    draw_t1err_curve(pv_thresholds, t1err, label, len(pv), gc_correct=gc_correct)


def compute_t1err_data(pv, i_causal):
    gc_correct = False
    pv_thresholds = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6] #, 5e-7, 1e-7, 5e-8, 1e-8]



    n_causal = i_causal.sum()

    #compute power and Type-1 error
    t1err = np.zeros_like(pv_thresholds)
    
    if gc_correct:
        t1err_corr = np.zeros_like(pv_thresholds)
        # compute lambda on all p-values?
        lambda_gc = estimate_lambda(pv)        
        pvcorr = stats.chi2.sf(stats.chi2.isf(pv,1)/lambda_gc,1)

    for i_t, t in enumerate(pv_thresholds):
        #compute uncorrected power and T1
        i_lower = pv<t
        t1err[i_t] = (~i_causal[i_lower]).sum()/(1.0*(len(i_causal)-n_causal))

        #compute GC corrected Power and T1
        if gc_correct:
            i_lower_corr = pvcorr<t
            t1err_corr[i_t] = (~i_causal[i_lower_corr]).sum()/(1.0*(len(i_causal)-n_causal))

    return pv_thresholds, t1err


def draw_t1err_curve(pv_thresholds, t1err, label, num_trials):
    import pylab
    pylab.plot(pv_thresholds, t1err, "-o", label=label)
    pylab.yscale("log")
    pylab.xscale("log")
    pylab.xlabel(r"$\alpha$",fontsize="large")
    pylab.ylabel("type I error",fontsize="large")
    pylab.grid(True)
    pylab.xlim([1e-6,1e-3])
    pylab.ylim([1e-6,1e0])

    rt = pv_thresholds[::-1]

    import scipy.stats as stats
    lower = [max(1e-7,(stats.distributions.binom.ppf(0.025, num_trials, t)-1)/float(num_trials)) for t in rt]
    upper = [stats.distributions.binom.ppf(0.975, num_trials, t)/float(num_trials) for t in rt]
    pylab.fill_between(rt, lower, upper, alpha=0.7, facecolor='#DDDDDD')
    pylab.plot(pv_thresholds, pv_thresholds, 'k--')
    pylab.legend(loc="lower right")


###############################################################
# Power
###############################################################

def plot_power_noshow(pv, i_causal, label="", gc_correct=False):
    pv_thresholds, power = compute_power_data(pv, i_causal)
    draw_power_curve(pv_thresholds, power, label, gc_correct=gc_correct)



def compute_power_data(pv, i_causal):


    pv_thresholds = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]

    n_causal = i_causal.sum()
    power = np.zeros(len(pv_thresholds), dtype=np.float)


    for i_t, t in enumerate(pv_thresholds):
        #compute uncorrected power and T1

        i_lower = pv<t
        power[i_t] = i_causal[i_lower].sum()/(1.0*(n_causal))


    return pv_thresholds, power


def draw_power_curve(pv_thresholds, power, label):
    import pylab
    pylab.plot(pv_thresholds, power, "-o", label=label)


    pylab.yscale("log")
    pylab.xscale("log")
    pylab.xlabel(r'$\alpha$', fontsize="large")
    pylab.ylabel("power", fontsize="large")
    pylab.grid(True)
    
    pylab.legend(loc="lower right")
        

###############################################################
# auROC
###############################################################

def plot_roc(y, out, label="", out_fn=None):
    """
    show or save ROC curve
    """

    import pylab
    pylab.figure()
    plot_roc_noshow(y, out, label=label)

    if not out_fn is None:
        pylab.savefig(out_fn)
    else:
        pylab.show()


def plot_roc_noshow(y, out, label=""):
    """
    create area under the receiver operator characteristic curve plot
    """

    fpr, tpr, roc_auc = compute_roc_data(y, out)
    draw_roc_curve(fpr, tpr, roc_auc, label)


def compute_roc_data(y, out):
    """
    compte relevant metrics for auROC
    """

    # plot auc
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, out)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def draw_roc_curve(fpr, tpr, roc_auc, label):
    
    if len(fpr) > 1000:
        sub_idx = [int(a) for a in np.linspace(0, len(fpr)-1, num=1000, endpoint=True)]
        fpr, tpr = fpr[sub_idx], tpr[sub_idx]

    import pylab
    pylab.plot(fpr, tpr, label='%s (area = %0.4f)' % (label, roc_auc))
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('Receiver operating characteristic example')
    pylab.grid(True)
    pylab.legend(loc="lower right")


###############################################################
# auPRC
###############################################################

def plot_prc(y, out, label="", out_fn=None):
    """
    plot precision recall plot
    """

    import pylab
    pylab.figure()
    plot_prc_noshow(y, out, label=label)

    if not out_fn is None:
        pylab.savefig(out_fn)
    else:
        pylab.show()


def plot_prc_noshow(y, out, label=""):
    precision, recall, area = compute_prc_data(y, out)
    draw_prc_curve(precision, recall, area, label)


def compute_prc_data(y, out):
    # Compute Precision-Recall and plot curve
    from sklearn.metrics import precision_recall_curve, auc

    precision, recall, thresholds = precision_recall_curve(y, out)
    area = auc(recall, precision)

    return precision, recall, area


def draw_prc_curve(precision, recall, area, label):
 
    import pylab
    pylab.plot(recall, precision, label='%s (area = %0.4f)' % (label, area))
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.ylim([0.0, 1.05])
    pylab.xlim([0.0, 1.0])
    pylab.grid(True)
    pylab.legend(loc="upper right")


def merge_results(results_dir, fn_filter_list, mindist):
    """
    visualize gwas results based on results file names
    """

    files = [fn for fn in os.listdir(results_dir) if fn.endswith("pickle")]

    import pylab
    pylab.figure()

    for fn_idx, fn_filter in enumerate(fn_filter_list):

        method_files = [fn for fn in files if fn.find(fn_filter) != -1]

        p_values = []
        p_values_lin = []
        i_causal = []

        for method_fn in method_files:
            tmp_fn = results_dir + "/" + method_fn
            print(tmp_fn)
            dat = load(tmp_fn)

            pv_m, i_causal_m = cut_snps_close_to_causals(dat["p_values_uncut"], dat["pos"], dat["causal_idx"], mindist=mindist)
            pv_lin_m, i_causal_m2 = cut_snps_close_to_causals(dat["p_values_lin_uncut"], dat["pos"], dat["causal_idx"], mindist=mindist)

            np.testing.assert_array_equal(i_causal_m, i_causal_m2)

            p_values.extend(pv_m)
            p_values_lin.extend(pv_lin_m)
            i_causal.extend(i_causal_m)

        p_values = np.array(p_values)
        p_values_lin = np.array(p_values_lin)
        i_causal = np.array(i_causal)


        method_label = fn_filter.replace("_", "")# underscore prefix hides label
        pylab.subplot(221)
        plot_prc_noshow(i_causal, -p_values, label=method_label)
        if fn_idx == 0:
            plot_prc_noshow(i_causal, -p_values_lin, label="lin")

        pylab.subplot(222)
        plot_roc_noshow(i_causal, -p_values, label=method_label)
        if fn_idx == 0:
            plot_roc_noshow(i_causal, -p_values_lin, label="lin")
                
        pylab.subplot(223)
        plot_t1err_noshow(p_values, i_causal, label=method_label)
        if fn_idx == 0:
            plot_t1err_noshow(p_values_lin, i_causal, label="lin")

        pylab.subplot(224)
        plot_power_noshow(p_values, i_causal, label=method_label)
        if fn_idx == 0:
            plot_power_noshow(p_values_lin, i_causal, label="lin")

        print(p_values)
        print(i_causal)


    pylab.show()


if __name__ == "__main__":
    num = 700000 * 500
    pv = X = np.random.random((num))
    i_causal = X = np.ones((num), dtype=np.bool)
    
    t0 = time.time()
    compute_power_data(pv, i_causal)
    print(("time taken:", time.time() - t0))
