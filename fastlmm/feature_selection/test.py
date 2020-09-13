from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import logging

from fastlmm.feature_selection import FeatureSelectionStrategy, load_snp_data
from pysnptools.snpreader import Bed
import pysnptools.util.pheno as pstpheno
import fastlmm.inference.linear_regression as lin_reg 
from pysnptools.snpreader import Hdf5
from pysnptools.snpreader import Dat
from pysnptools.snpreader import Ped

#  sklearn
from sklearn.model_selection import KFold
from pysnptools.standardizer import Unit

from fastlmm.inference import getLMM
import unittest
import os.path

import pysnptools.util
from fastlmm.feature_selection.feature_selection_two_kernel import FeatureSelectionInSample
import fastlmm.util.standardizer as stdizer
from six.moves import range



class TestTwoKernelFeatureSelection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.snp_fn = currentFolder + "/../../tests/datasets/mouse/alldata"
        self.pheno_fn = currentFolder + "/../../tests/datasets/mouse/pheno_10_causals.txt"
        #self.cov_fn = currentFolder + "/examples/toydata.cov"

        # load data
        ###################################################################
        snp_reader = Bed(self.snp_fn,count_A1=False)
        pheno = pstpheno.loadOnePhen(self.pheno_fn)
        #cov = pstpheno.loadPhen(self.cov_fn)
        
        # intersect sample ids
        snp_reader, pheno = pysnptools.util.intersect_apply([snp_reader, pheno])
        
        self.G = snp_reader.read(order='C').val
        self.G = stdizer.Unit().standardize(self.G)
        self.G.flags.writeable = False
        self.y = pheno['vals'][:,0]
        self.y.flags.writeable = False

        # load pcs
        #self.G_cov = cov['vals']
        self.G_cov = np.ones((len(self.y), 1))
        self.G_cov.flags.writeable = False
        

    def test_regression_lmm(self):


        # invoke fs
        select = FeatureSelectionInSample(n_folds=2, max_log_k=6, order_by_lmm=True, measure="mse", random_state=42)
        best_k, feat_idx, best_mix, best_delta = select.run_select(self.G, self.G, self.y, cov=self.G_cov)    
    
        # print results
        print("best_k:", best_k)
        print("best_mix:", best_mix)
        print("best_delta:", best_delta)

        self.assertEqual(best_k, 64)
        self.assertAlmostEqual(best_mix, 0.8621642030968627, places=6)
        self.assertAlmostEqual(best_delta, 0.7255878551207211, places=6)


    def test_regression_lr(self):

        # invoke fs
        select = FeatureSelectionInSample(n_folds=2, max_log_k=6, order_by_lmm=False, measure="mse", random_state=42)
        best_k, feat_idx, best_mix, best_delta = select.run_select(self.G, self.G, self.y, cov=self.G_cov)    
    
        # print results
        print("best_k:", best_k)
        print("best_mix:", best_mix)
        print("best_delta:", best_delta)

        self.assertEqual(best_k, 32)
        self.assertAlmostEqual(best_mix, 0.6786566031577429, places=6)
        self.assertAlmostEqual(best_delta, 0.70148446599200931, places=6)


class TestFeatureSelection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.snpreader_bed = Bed(currentFolder + "/examples/toydata.5chrom.bed",count_A1=False)
        #Creating Hdf5 data ...
        #snpData = self.snpreader_bed.read()
        #Hdf5.write(snpData, currentFolder + "/examples/toydata.snpmajor.hdf5")
        #Hdf5.write(snpData, currentFolder + "/examples/toydata.iidmajor.hdf5",snp_major=False)

        #Creating Dat data ...
        #snpData = self.snpreader_bed.read()
        #Dat.write(snpData, currentFolder + "/examples/toydata.dat")

        ###Creating Ped data ...
        #snpData = self.snpreader_bed.read()
        #Ped.write(snpData, currentFolder + "/examples/toydata.ped")
        #fromPed = Ped(currentFolder + "/examples/toydata").read()
        #self.assertTrue(np.allclose(snpData.val, fromPed.val, rtol=1e-05, atol=1e-05))
        
        
        self.snpreader_hdf5 = Hdf5(currentFolder + "/examples/toydata.snpmajor.hdf5")
        self.snpreader_dat = Dat(currentFolder + "/examples/toydata.dat")
        self.snpreader_ped = Ped(currentFolder + "/examples/toydata")
        self.pheno_fn = currentFolder + "/examples/toydata.phe"
        self.pheno_shuffleplus_fn = currentFolder + "/examples/toydata.shufflePlus.phe"

 

    """
    make sure the used pca yields the same result as standard pca
    def test_pca(self):

        from sklearn.decomposition import PCA, KernelPCA

        print "testing PCA"

        num_steps_delta = 5
        num_folds = 2
        num_pcs = 2
        random_state = 42
        output_prefix = None
        fss = FeatureSelectionStrategy(self.snpreader, self.pheno_fn, num_folds, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=20000,count_A1=False)
        fss.run_once()

        pca = PCA(n_components=fss.num_pcs)
        pcs = pca.fit_transform(fss.G)

        for i in xrange(num_pcs):
            
            pc_1 = fss.pcs[:,i]
            pc_2 = pcs[:,i]
            # sign -1 if signs different, 1 else
            sign = np.sign(pc_1[0]) * np.sign(pc_2[0])

            np.testing.assert_array_almost_equal(pc_1, sign*pc_2)
    """
 

    def test_regression_bed(self):
        self.regression(self.snpreader_bed, self.regular_regression_answers)

    def test_regression_hdf5(self):
        self.regression(self.snpreader_hdf5, self.regular_regression_answers)

    def test_regression_dat(self):
        self.regression(self.snpreader_dat, self.regular_regression_answers)

    def test_regression_ped(self):
        self.regression(self.snpreader_ped, self.regular_regression_answers)

    regular_regression_answers = (22, 20.085536923, 61.2448170241, 0.67586545761317196)
    cov_pca_regression_answers = (22, 20.085536923, 61.8146293815, 0.692761716513)
    cov_pca_insample_cv_regression_answers = (22, 1.6513737331988527, 63.6062289765, 0.71724685708485092)

    def test_regression_cov_pcs(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        cov_fn = currentFolder + "/examples/toydata.cov"
        self.regression(self.snpreader_bed, self.cov_pca_regression_answers, cov_fn=cov_fn, num_pcs=3)

    def test_regression_cov_pcs_insample_cv(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        cov_fn = currentFolder + "/examples/toydata.cov"
        self.regression(self.snpreader_bed, self.cov_pca_insample_cv_regression_answers, cov_fn=cov_fn, num_pcs=3, strategy = "insample_cv", delta=5)

    def regression(self, snpreader, answers, cov_fn=None, num_pcs=0, strategy = "lmm_full_cv", delta=7):
        """
        compare against previous results of this code base
        """
    
        # set up grid
        ##############################
        num_steps_delta = 5
        num_steps_k = 5
        num_folds = 2


        # log_2 space and all SNPs
        k_values = np.array(np.logspace(0, 9, base=2, num=num_steps_k, endpoint=True),dtype=np.int64).tolist() + [10000]
        #k_values = np.logspace(0, 9, base=2, num=num_steps_k, endpoint=True).tolist() + [10000]
        delta_values = np.logspace(-3, 3, endpoint=True, num=num_steps_delta, base=np.exp(1))
        random_state = 42

        output_prefix = None

        # select by LL
        fss = FeatureSelectionStrategy(snpreader, self.pheno_fn, num_folds, random_state=random_state, cov_fn=cov_fn, num_pcs=num_pcs, interpolate_delta=True,count_A1=False)
        best_k, best_delta, best_obj, best_snps = fss.perform_selection(k_values, delta_values, strategy, output_prefix=output_prefix, select_by_ll=True,create_pdf=False)
        
        self.assertEqual(best_k, answers[0])
        self.assertAlmostEqual(best_delta, answers[1], delta)
        self.assertTrue(abs(best_obj-answers[2])<.005) #accept a range answers for when standardization is done with doubles, floats, etc

        # select by MSE
        fss = FeatureSelectionStrategy(snpreader, self.pheno_fn, num_folds, random_state=random_state, cov_fn=cov_fn, num_pcs=num_pcs, interpolate_delta=True,count_A1=False)
        best_k, best_delta, best_obj, best_snps = fss.perform_selection(k_values, delta_values, strategy, output_prefix=output_prefix, select_by_ll=False,create_pdf=False)
        
        self.assertEqual(best_k, answers[0])
        self.assertAlmostEqual(best_delta, answers[1], delta)
        self.assertAlmostEqual(best_obj, answers[3])

    def test_blocking_bed(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.blocking(currentFolder + "/examples/toydata.5chrom.bed") # use string instead of reader, to test that strings work

    def test_blocking_hdf5(self):
        self.blocking(self.snpreader_hdf5)

    def test_blocking_dat(self):
        self.blocking(self.snpreader_dat)

    def test_blocking_ped(self):
        self.blocking(self.snpreader_ped)


    tolerance = 1e-4

    def test_blocking_cov_pcs(self):
        self.blocking_cov_pcs(strategy="lmm_full_cv")

    def getworkingtest_blocking_cov_pcs_insample_cv(self): #!!!
        self.blocking_cov_pcs(strategy="insample_cv")

    @staticmethod
    def reference_file(outfile):
        #!!similar code elsewhere
        import platform;
        os_string=platform.platform()

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..","tests")
        windows_fn = file_path + '/expected-Windows/'+outfile
        assert os.path.exists(windows_fn), "Can't find file '{0}'".format(windows_fn)
        debian_fn = file_path + '/expected-debian/'+outfile
        if not os.path.exists(debian_fn): #If reference file is not in debian folder, look in windows folder
            debian_fn = windows_fn

        if "debian" in os_string or "Linux" in os_string:
            if "Linux" in os_string:
                logging.warning("comparing to Debian output even though found: %s" % os_string)
            return debian_fn
        else:
            if "Windows" not in os_string:
                logging.warning("comparing to Windows output even though found: %s" % os_string)
            return windows_fn 


    def blocking_cov_pcs(self,strategy):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        cov_fn = currentFolder + "/examples/toydata.cov"
        output_dir="tmp"
        try:
            os.mkdir(output_dir)
        except:
            pass
        output_dir = output_dir + "/feature_selection"
        try:
            os.mkdir(output_dir)
        except:
            pass
        self.blocking(self.snpreader_bed, cov_fn, num_pcs=3, strategy=strategy, output_prefix=os.path.join(output_dir,strategy))

        for outfile in os.listdir(output_dir):
            referenceOutfile = TestFeatureSelection.reference_file(outfile)

            import fastlmm.util.util as ut
            if outfile.lower().endswith(".pdf") or outfile == "output_prefix_report.txt" or outfile.lower().endswith("_k_pcs.txt"):
                self.assertTrue(os.path.exists(referenceOutfile))
            else:
                delimiter = "," if outfile.lower().endswith(".csv") else "\t"
                tmpOutfile=os.path.join(output_dir,outfile)
                out,msg=ut.compare_mixed_files(tmpOutfile, referenceOutfile, self.tolerance,delimiter=delimiter)
                #if not out:
                #    print "False" #import pdb; pdb.set_trace() #This will mess up LocalMultiProc runs
                self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))
      
     
    def blocking(self, snpreader, cov_fn=None, num_pcs=0, output_prefix = None, strategy="lmm_full_cv"):
        """
        compare three different cases:

        To control memory use, we've introduced a parameter called "num_snps_in_memory", which defaults to 100000. 
        Here are the interesting cases to consider (and choose num_snps_in_memory accordingly):

        1) num_snps_in_memory > total_num_snps

           In this case, the same code as before should be 
           executed (except the kernel matrix on all SNPs is now cached). 


        2) num_snps_in_memory < total_num_snps
            num_snps_in_memory > k (excluding all_snps)

            Here, the linear regression will be blocked, 
            while the data for cross-validation is cached, 
            saving time for loading and re-indexing.


        3) num_snps_in_memory < total_num_snps
            num_snps_in_memory < k (excluding all_snps)

            Finally, both operations - linear regression 
            and building the kernel will be blocked.

        4,5,6) Same as #1,2,3, but with a phenos that has extra iids and for which the iids are shuffled.


        """

        # set up grid
        ##############################
        num_steps_delta = 5
        num_folds = 2

        # log_2 space and all SNPs
        k_values = [0, 1, 5, 10, 100, 500, 700, 10000] 
        delta_values = np.logspace(-3, 3, endpoint=True, num=num_steps_delta, base=np.exp(1))
        
        random_state = 42


        # case 1
        fss_1 = FeatureSelectionStrategy(snpreader, self.pheno_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=20000,count_A1=False)
        best_k_1, best_delta_1, best_obj_1, best_snps_1 = fss_1.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        #some misc testing
        from fastlmm.feature_selection import PerformSelectionDistributable as psd
        perform_selection_distributable = psd.PerformSelectionDistributable(fss_1, k_values, delta_values, strategy, output_prefix, select_by_ll=True, penalty=0.0,create_pdf=False)
        self.assertEqual(perform_selection_distributable.work_count, 3)
        s = perform_selection_distributable.tempdirectory
        s = str(perform_selection_distributable)
        s = "%r" % perform_selection_distributable
        from fastlmm.feature_selection.feature_selection_cv import GClass
        s = "%r" % GClass.factory(snpreader,1000000, Unit(), 50,count_A1=False)
        s = s
        #!!making  test for each break point.


        # case 2
        fss_2 = FeatureSelectionStrategy(snpreader, self.pheno_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=5000,count_A1=False)
        best_k_2, best_delta_2, best_obj_2, best_snps_2 = fss_2.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        # case 3
        fss_3 = FeatureSelectionStrategy(snpreader, self.pheno_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=600,count_A1=False)
        best_k_3, best_delta_3, best_obj_3, best_snps_3 = fss_3.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        # case 4
        fss_4 = FeatureSelectionStrategy(snpreader, self.pheno_shuffleplus_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=20000,count_A1=False)
        best_k_4, best_delta_4, best_obj_4, best_snps_4 = fss_4.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        # case 5
        fss_5 = FeatureSelectionStrategy(snpreader, self.pheno_shuffleplus_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=5000,count_A1=False)
        best_k_5, best_delta_5, best_obj_5, best_snps_5 = fss_5.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        # case 6
        fss_6 = FeatureSelectionStrategy(snpreader, self.pheno_shuffleplus_fn, num_folds, cov_fn=cov_fn, random_state=random_state, num_pcs=num_pcs, interpolate_delta=True, num_snps_in_memory=600,count_A1=False)
        best_k_6, best_delta_6, best_obj_6, best_snps_6 = fss_6.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=True, strategy=strategy,create_pdf=False)

        self.assertEqual(int(best_k_1), int(best_k_2))
        self.assertEqual(int(best_k_1), int(best_k_3))
        #self.assertEqual(int(best_k_1), int(best_k_4))
        #self.assertEqual(int(best_k_1), int(best_k_5))
        #self.assertEqual(int(best_k_1), int(best_k_6))
        self.assertAlmostEqual(best_obj_1, best_obj_2)
        self.assertAlmostEqual(best_obj_1, best_obj_3)
        #self.assertAlmostEqual(best_obj_1, best_obj_4)
        self.assertAlmostEqual(best_obj_4, best_obj_5)
        self.assertAlmostEqual(best_obj_4, best_obj_6)

        if strategy != "insample_cv":
            self.assertAlmostEqual(best_delta_1, best_delta_2)
            self.assertAlmostEqual(best_delta_1, best_delta_3)
            #self.assertAlmostEqual(best_delta_1, best_delta_4)
            self.assertAlmostEqual(best_delta_4, best_delta_5)
            self.assertAlmostEqual(best_delta_4, best_delta_6)
       

    def test_log_likelihood_bed(self):
        self.log_likelihood(self.snpreader_bed)
        
    def test_log_likelihood_hdf5(self):
        self.log_likelihood(self.snpreader_hdf5)

    def test_log_likelihood_dat(self):
        self.log_likelihood(self.snpreader_dat)

    def test_log_likelihood_ped(self):
        self.log_likelihood(self.snpreader_ped)

    def log_likelihood(self, snpreader):
        """
        test mean, variance against C++ results (autoselect):

        FastLmmC.exe -autoselect test -bfilesim toydata -pheno toydata.phe -logDelta X -autoSelectSearchValues k -verboseOutput > debug.txt
        """

        # low-rank, small delta
        ll_1_expected = np.array([79.7217, 80.5289, 79.9218, 53.7485, 71.4134, 71.5751, 58.4209, 69.82, 85.5727, 72.7218])
        ll_1 = core_run(snpreader, self.pheno_fn, 50, np.exp(-5))

        for i in range(len(ll_1)):
            np.testing.assert_approx_equal(ll_1[i], ll_1_expected[i], significant=3, err_msg='Log-likelihoods differ', verbose=True)


        # low-rank, large delta
        ll_2_expected = np.array([68.7098, 70.0446, 75.1816, 62.3675, 69.34, 74.6755, 59.9937, 66.6408, 74.1564, 68.3146])
        ll_2 = core_run(snpreader, self.pheno_fn, 50, np.exp(10))

        for i in range(len(ll_2)):
            np.testing.assert_approx_equal(ll_2[i], ll_2_expected[i], significant=3, err_msg='Log-likelihoods differ', verbose=True)


        # full-rank, small delta
        # these values appear to indicate numerical instability. the python version will continue to match these
        # results for the time being, but provide a flag "robust" to lead to a numerically more stable solution
        ll_3_expected = np.array([1636.33, 28711.8, 32008.8, 1363.74, 128444, 22277.6, 16389.2, 95458.7, 4710.33, 68308.9])
        ll_3 = core_run(snpreader, self.pheno_fn, 5000, np.exp(-5))

        for i in range(len(ll_3)):
            np.testing.assert_approx_equal(ll_3[i], ll_3_expected[i], significant=2, err_msg='Log-likelihoods differ', verbose=True)

        
        # full-rank, small delta
        ll_4_expected = np.array([68.4794, 70.7483, 76.6886, 62.2721, 69.0659, 76.8207, 59.5216, 66.1517, 75.9149, 68.6061])
        ll_4 = core_run(snpreader, self.pheno_fn, 5000, np.exp(10))

        for i in range(len(ll_4)):
            np.testing.assert_approx_equal(ll_4[i], ll_4_expected[i], significant=3, err_msg='Log-likelihoods differ', verbose=True)


        # square matrix, delta 2
        ll_5_expected = np.array([259.91, 351.914, 389.971, 313.807, 334.883, 345.312, 221.556, 206.15, 270.53, 304.231])
        ll_5 = core_run(snpreader, self.pheno_fn, 500, np.exp(2))

        for i in range(len(ll_5)):
            np.testing.assert_approx_equal(ll_5[i], ll_5_expected[i], significant=3, err_msg='Log-likelihoods differ', verbose=True)

def core_run(snpreader, pheno_fn, k, delta):
    """
    extracted core functionality, to avoid shuffle of data and not correct delta
    """

    G, X, y = load_snp_data(snpreader, pheno_fn, standardizer=Unit())
    kf = KFold(n_splits=10, shuffle=False).split(list(range(len(y))))

    ll = np.zeros(10)

    fold_idx = 0
    fold_data = {}
    for split_idx, (train_idx, test_idx) in enumerate(kf):
        fold_idx += 1

        fold_data["train_idx"] = train_idx
        fold_data["test_idx"] = test_idx

        # set up data
        ##############################
        fold_data["G_train"] = G[train_idx,:].read()
        fold_data["G_test"] = G[test_idx,:]

        fold_data["X_train"] = X[train_idx]
        fold_data["X_test"] = X[test_idx]

        fold_data["y_train"] = y[train_idx]
        fold_data["y_test"] = y[test_idx]


        # feature selection
        ##############################
        _F,_pval = lin_reg.f_regression_block(lin_reg.f_regression_cov_alt,fold_data["G_train"].val,fold_data["y_train"],blocksize=1E4,C=fold_data["X_train"])
        feat_idx = np.argsort(_pval)
        fold_data["feat_idx"] = feat_idx
        
        # re-order SNPs (and cut to max num)
        ##############################
        fold_data["G_train"] = fold_data["G_train"][:,feat_idx[0:k]].read()
        fold_data["G_test"] = fold_data["G_test"][:,feat_idx[0:k]].read()

        model = getLMM()
        model.setG(fold_data["G_train"].val)
        model.sety(fold_data["y_train"])
        model.setX(fold_data["X_train"])

        REML = False
        
        # predict on test set
        res = model.nLLeval(delta=delta, REML=REML)
        model.setTestData(Xstar=fold_data["X_test"], G0star=fold_data["G_test"].val)
        model.predictMean(beta=res["beta"], delta=delta)
        #mse_cv1[k_idx, delta_idx] = mean_squared_error(fold_data["y_test"],
        #out)
        ll[split_idx] = model.nLLeval_test(fold_data["y_test"], res["beta"], sigma2=res["sigma2"], delta=delta)


    return ll


def getTestSuite():
    """
    set up composite test suite
    """
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTwoKernelFeatureSelection)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestFeatureSelection)
    return unittest.TestSuite([suite1,suite2])


if __name__ == '__main__':
    unittest.main()
