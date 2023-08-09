from __future__ import absolute_import
from __future__ import print_function
import logging
import numpy as np
import unittest
import os.path
import doctest
import pandas as pd
import pysnptools.util as pstutil
import sys

from fastlmm.inference import LinearRegression
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno, SnpData
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.standardizer import Unit, Standardizer
from pysnptools.standardizer import Identity as SS_Identity
from pysnptools.kernelstandardizer import Identity as KS_Identity
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData, SnpKernel
from six.moves import range

class TestLinRegTrain(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))

        self.snpreader_whole = Bed(self.pythonpath + "/tests/datasets/synth/all.bed",count_A1=False)
        self.covariate_whole = Pheno(self.pythonpath + "/tests/datasets/synth/cov.txt")
        self.pheno_whole = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt")

    tempout_dir = "tempout/linear_regression"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".dat")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_lr_real(self):
        do_plot = False

        import pylab
        logging.info("TestLinRegTrain test_lr_real")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        #make covar just numbers 0,1,...
        covar = self.covariate_whole.read()
        covar.val = np.array([[float(num)] for num in range(covar.iid_count)])
        covariate_train = covar[train_idx,:].read()
        covariate_test = covar[test_idx,:].read()
        K0_test_test = KernelIdentity(covariate_test.iid)

        #make pheno  # pheno = 2*covar+100+normal(0,1)*10
        pheno = self.pheno_whole.read()
        np.random.seed(0)
        pheno.val = covar.val * 2.0 + 100 + np.random.normal(size=covar.val.shape)*10

        pheno_train = pheno[train_idx,:].read()
        pheno_test = pheno[test_idx,:].read()

        if do_plot:
            #Plot training x and y, testing x and y
            pylab.plot(covariate_train.val, pheno_train.val,".",covariate_test.val, pheno_test.val,".")
            pylab.suptitle("Plot training x and y, testing x and y")
            pylab.show()

        Xtrain = np.c_[covariate_train.val,np.ones((covariate_train.iid_count,1))]
        Xtest = np.c_[covariate_test.val,np.ones((covariate_test.iid_count,1))]
        lsqSol = np.linalg.lstsq(Xtrain, pheno_train.val[:,0],rcond=-1)
        bs=lsqSol[0] #weights
        r2=lsqSol[1] #squared residuals
        D=lsqSol[2]  #rank of design matrix
        N=pheno_train.iid_count
        REML = False
        if not REML:
            sigma2 = float(r2/N)
            nLL =  N*0.5*np.log(2*np.pi*sigma2) + N*0.5
        else:
            sigma2 = float(r2 / (N-D))
            nLL = N*0.5*np.log(2*np.pi*sigma2) + 0.5/sigma2*r2;
            nLL -= 0.5*D*np.log(2*np.pi*sigma2);#REML term

        predicted = Xtest.dot(bs)
        yerr = [np.sqrt(sigma2)] * len(predicted)
        if do_plot:
            pylab.plot(covariate_test.val, pheno_test.val,"g.",covariate_test.val, predicted,"r.")
            pylab.xlim([-1, 10])
            pylab.errorbar(covariate_test.val, predicted,yerr,linestyle='None')
            pylab.suptitle("real linear regression: actual to prediction")
            pylab.show()

        #These should all give the same result
        first_name = None
        for name,K0_train,K0_whole_test in [("Identity Kernel",None,None)]:

            first_name = first_name or name
            #Learn model, save, load
            modelx = LinearRegression().fit(K0_train=K0_train, X=covariate_train, y=pheno_train)
                
                
            filename = self.tempout_dir + "/model_lr_real.flm.p"
            pstutil.create_directory_if_necessary(filename)
            model = modelx

            do_test_on_train = True
            if do_test_on_train:
                #Predict with model (test on train)
                predicted_pheno, covar = model.predict(K0_whole_test=K0_train, X=covariate_train) #test on train
                output_file = self.file_name("lr_reala_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr_reala.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covariate_train.val, pheno_train.val,"g.",covariate_train.val, predicted,"r.")
                    pylab.xlim([0, 50])
                    pylab.ylim([100, 200])
                    pylab.errorbar(covariate_train.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_pheno,"lr2a_"+first_name)
                self.compare_files(covar2,"lr2a.cov_"+first_name)

            #Predict with model (test on test)
            predicted_pheno, covar  = model.predict(K0_whole_test=K0_whole_test, X=covariate_test) #test on train
            output_file = self.file_name("lr_realb_"+name)
            Dat.write(output_file,predicted_pheno)
            covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
            output_file = self.file_name("lr_realb.cov_"+name)
            Dat.write(output_file,covar2)

            yerr = np.sqrt(np.diag(covar.val))
            predicted = predicted_pheno.val
            if do_plot:
                pylab.plot(covariate_test.val, pheno_test.val,"g.",covariate_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covariate_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                pylab.show()
                ## Plot y and predicted y (test on train)
                #pylab.plot(pheno_test.val,predicted_pheno.val,".")
                #pylab.suptitle(name+": test on test: true target to prediction")
                #pylab.show()

            self.compare_files(predicted_pheno,"lr2b_"+first_name)
            self.compare_files(covar2,"lr2b.cov_"+first_name)



    def compare_files(self,answer,ref_base):
        reffile = TestFeatureSelection.reference_file("fastlmm/"+ref_base+".dat") #Uses same results folder as lmm_train
        reference=Dat(reffile).read()
        assert np.array_equal(answer.col,reference.col), "sid differs. File '{0}'".format(reffile)
        assert np.array_equal(answer.row,reference.row), "iid differs. File '{0}'".format(reffile)
        for iid_index in range(reference.row_count):
            for sid_index in range(reference.col_count):
                a_v = answer.val[iid_index,sid_index]
                r_v = reference.val[iid_index,sid_index]
                assert abs(a_v - r_v) < 1e-4, "Value at {0},{1} differs too much from file '{2}'".format(iid_index,sid_index,reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.inference.linear_regression'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestLinRegTrain)
    return unittest.TestSuite([suite1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # this import is needed for the runner
    #from fastlmm.inference.tests.test_linear_regression import TestLinRegTrain
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else: #Cluster test run
        from pysnptools.util.mapreduce1.distributabletest import DistributableTest

        runner = HPC(10, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond',
                     remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
                     update_remote_python_parent=True,
                     priority="AboveNormal",mkl_num_threads=1)
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
