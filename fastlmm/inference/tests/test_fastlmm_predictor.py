from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
#import matplotlib
#matplotlib.use("TKAgg",warn=False)
#import pylab

import logging
import numpy as np
import unittest
import os.path
import doctest
import pandas as pd
import sys
import pysnptools.util as pstutil

from fastlmm.inference import FastLMM
from fastlmm.inference.fastlmm_predictor import _SnpWholeTest
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno, SnpData
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.standardizer import Unit, Standardizer
from pysnptools.standardizer import Identity as SS_Identity
from pysnptools.kernelstandardizer import Identity as KS_Identity
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData, SnpKernel
from pysnptools.standardizer import DiagKtoN

class TestFastLMM(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))

        self.snpreader_whole = Bed(self.pythonpath + "/tests/datasets/synth/all.bed",count_A1=False)
        self.covariate_whole = Pheno(self.pythonpath + "/tests/datasets/synth/cov.txt")
        self.pheno_whole = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt")

    tempout_dir = "tempout/fastlmm"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".dat")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_api(self):
        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        #####################################################
        # Train and standardize cov and then apply to test
        #####################################################

        cov_train, unit_trained = self.covariate_whole[train_idx,:].read().standardize(Unit(),return_trained=True)
        cov_test = self.covariate_whole[test_idx,:].read().standardize(unit_trained)

        #####################################################
        # standardize whole kernel from snps (both ways) and then pull out the 3 parts
        #####################################################
        
        whole_kernel = SnpKernel(self.covariate_whole,Unit()).read().standardize(DiagKtoN())
        train_kernel = whole_kernel[train_idx].read(order='A',view_ok=True)
        test_kernel = whole_kernel[train_idx,test_idx].read(order='A',view_ok=True)
        test_test_kernel = whole_kernel[test_idx,test_idx].read(order='A',view_ok=True)

        #####################################################
        # create train_train, train_test, and test_test based on just the training snps (both standardizations)
        #####################################################

        K_train = SnpKernel(self.snpreader_whole[train_idx,:],Unit(),block_size=100)
        train_train_kernel, snp_trained, kernel_trained = K_train._read_with_standardizing(to_kerneldata=True, kernel_standardizer=DiagKtoN(), return_trained=True)

        K_whole_test = _SnpWholeTest(train=self.snpreader_whole[train_idx,:],test=self.snpreader_whole[test_idx,:],standardizer=snp_trained,block_size=100)
        train_idx2 = K_whole_test.iid0_to_index(self.snpreader_whole.iid[train_idx]) #The new reader may have the iids in a different order than the original reader
        train_test_kernel = K_whole_test[train_idx2,:].read().standardize(kernel_trained)

        test_idx2 = K_whole_test.iid0_to_index(self.snpreader_whole.iid[test_idx])
        test_test_kernel = K_whole_test[test_idx2,:].read().standardize(kernel_trained)

        #####################################################
        # How does predict look with whole_test as input?
        #####################################################

        # a. - standardize whole up front
        whole_kernel = SnpKernel(self.snpreader_whole,Unit(),block_size=100).read().standardize()
        train_kernel = whole_kernel[train_idx].read(order='A',view_ok=True)
        whole_test_kernel = whole_kernel[:,test_idx].read(order='A',view_ok=True)
        fastlmm1 = FastLMM(snp_standardizer=SS_Identity(), kernel_standardizer=KS_Identity())
        fastlmm1.fit(K0_train=train_kernel, X=self.covariate_whole, y=self.pheno_whole) #iid intersection means we won't really be using whole covar or pheno
        predicted_pheno, covar = fastlmm1.predict(K0_whole_test=whole_test_kernel, X=self.covariate_whole,count_A1=False)
        output_file = self.file_name("whole")
        Dat.write(output_file,predicted_pheno)
        self.compare_files(predicted_pheno,"whole")

        # b -- just files
        fastlmm2 = FastLMM()
        fastlmm2.fit(K0_train=self.snpreader_whole[train_idx,:], X=self.covariate_whole, y=self.pheno_whole[train_idx,:]) #iid intersection means we won't really be using whole covar
        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=self.snpreader_whole[test_idx,:], X=self.covariate_whole,count_A1=False)
        self.compare_files(predicted_pheno,"one")

    def test_notebook1(self):
        do_plot=False

        import matplotlib.pyplot as plt
        from pysnptools.snpreader import Pheno,Bed
        bed = Bed(self.pythonpath + "/tests/datasets/synth/all.bed",count_A1=False)
        cov = Pheno(self.pythonpath + "/tests/datasets/synth/cov.txt")
        pheno = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt").read()

        # Now we learn from the first 400 students.
        training = bed[:400,:] #!!!later: the learning code doesn't like it if there are two instances of bed[:400] that are not "is -equal"
        fastlmm2 = FastLMM(GB_goal=2).fit(K0_train=training,
                                            X=cov[:400,:],
                                            y=pheno[:400,:])

        # Predict on training data:
        predicted_score,covariance = fastlmm2.predict(K0_whole_test=training,
                                                            X=cov[:400,:],count_A1=False)

        assert np.array_equal(pheno.iid[:400],predicted_score.iid), "for plots to make sense, the iids must be in the order"
        if do_plot:
            plt.plot(pheno.val[:400,:],predicted_score.val,"b.",[-5,5],[-5,5],"-r")
            plt.errorbar(pheno.val[:400,:],predicted_score.val, yerr=np.sqrt(np.diag(covariance.val)),fmt='.')
            plt.xlabel('score (actual train)')
            plt.ylabel('predicted (test on train with stdev)')
            plt.show()

        # How well does this model predict the (unseen) TEST data?
        predicted_score,covariance = fastlmm2.predict(K0_whole_test=bed[400:500,:],
                                                            X=cov[400:500,:],count_A1=False)

        assert np.array_equal(pheno.iid[400:500],predicted_score.iid), "for plots to make sense, the iids must be in the order"
        if do_plot:
            plt.plot(pheno.val[400:500,:],predicted_score.val,"b.",[-5,5],[-5,5],"-r")
            plt.errorbar(pheno.val[400:500,:],predicted_score.val, yerr=np.sqrt(np.diag(covariance.val)),fmt='.')
            plt.xlabel('score (actual test)')
            plt.ylabel('predicted')
            plt.show()

    def test_one(self):
        logging.info("TestLmmTrain test_one")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        covariate_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]

        fastlmm1 = FastLMM(GB_goal=2).fit(K0_train=G0_train, X=covariate_train, y=pheno_train)
        filename = self.tempout_dir + "/model_one.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1
                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covariate_test = self.covariate_whole[test_idx,:]

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=G0_test, X=covariate_test,count_A1=False)

        output_file = self.file_name("one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one")

    def test_str(self):
        logging.info("TestLmmTrain test_str")

        G0_train = self.pythonpath + "/tests/datasets/synth/all.bed"
        covariate_train = None
        pheno_train = self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt"

        fastlmm1 = FastLMM(GB_goal=2).fit(K0_train=G0_train, X=covariate_train, y=pheno_train,count_A1=False)
        filename = self.tempout_dir + "/model_str.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1
                
        # predict on same
        G0_test = G0_train
        covariate_test = covariate_train

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=G0_test, X=covariate_test,count_A1=False)

        output_file = self.file_name("str")
        Dat.write(output_file,predicted_pheno)

        #pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"str")

    def test_lr_no_K0(self):
        logging.info("TestLinRegTrain test_lr_no_k0")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        covariate_train3 = self.covariate_whole[train_idx,:].read()
        covariate_train3.val = np.array([[float(num)] for num in range(covariate_train3.iid_count)])
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        np.random.seed(0)
        pheno_train3.val = covariate_train3.val * 2.0 + 100 + np.random.normal(size=covariate_train3.val.shape) # y = 2*x+100+normal(0,1)

        #Learn model, save, load
        fastlmm3x = FastLMM(GB_goal=2).fit(X=covariate_train3, y=pheno_train3)
        filename = self.tempout_dir + "/model3.flm.p"
        fastlmm3 = fastlmm3x

        #Predict with model (test on train)
        predicted_pheno, covariance = fastlmm3.predict(K0_whole_test=KernelIdentity(pheno_train3.iid), X=covariate_train3,count_A1=False) #test on train
        output_file = self.file_name("lr_no_k0")
        Dat.write(output_file,predicted_pheno)

        self.compare_files(predicted_pheno,"lr_no_k0")

    def test_lr_as_lmm(self):
            do_plot = False
            #later why does this test case generate two intersect info messages instead of just one?
            import pylab
            logging.info("TestLmmTrain test_lr_as_lmm")

            ###############################################################
            # Create a linear data set with just a little noise
            ###############################################################

            train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
            test_idx  = np.r_[0:10] # the first 10 iids

            #make covar just numbers 0,1,...
            covar = self.covariate_whole.read()
            covar.val = np.array([[float(num)] for num in range(covar.iid_count)])
            covar._name = 'np.array([[float(num)] for num in xrange(covar.iid_count)])'
            covariate_train = covar[train_idx,:].read()
            covariate_test = covar[test_idx,:].read()


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

            ###############################################################
            # Show that linear regression does a good job predicting
            ###############################################################

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

            ###############################################################
            # Use LMM as LR and apply test on train
            ###############################################################
            for force_full_rank in [True, False]:
                #Learn model, save, load
                fastlmmx = FastLMM(GB_goal=2,force_full_rank=force_full_rank).fit(K0_train=covariate_train, X=None, y=pheno_train)
                
                
                filename = self.tempout_dir + "/model_lr_as_lmm.flm.p"
                pstutil.create_directory_if_necessary(filename)
                fastlmm = fastlmmx

                do_test_on_train = True
                if do_test_on_train:
                    #Predict with model (test on train)
                    predicted_pheno, covar = fastlmm.predict(K0_whole_test=covariate_train, X=None,count_A1=False) #test on train
                    output_file = self.file_name("lr_as_lmma_")
                    Dat.write(output_file,predicted_pheno)
                    covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                    output_file = self.file_name("lr_as_lmma.cov_")
                    Dat.write(output_file,covar2)

                    yerr = np.sqrt(np.diag(covar.val))
                    predicted = predicted_pheno.val
                    if do_plot:
                        pylab.plot(covariate_train.val, pheno_train.val,"g.",covariate_train.val, predicted,"r.")
                        pylab.xlim([0, 50])
                        pylab.ylim([100, 200])
                        pylab.errorbar(covariate_train.val, predicted,yerr,linestyle='None')
                        pylab.suptitle("test on train: train X to true target (green) and prediction (red)")
                        pylab.show()

                    self.compare_files(predicted_pheno,"lr_as_lmma_")
                    self.compare_files(covar2,"lr_as_lmma.cov_")

                ###############################################################
                # Use LMM as LR and apply test on test
                ###############################################################

                #Predict with model (test on test)
                predicted_pheno, covar  = fastlmm.predict(K0_whole_test=covariate_test, X=None,count_A1=False) #test on train
                output_file = self.file_name("lr_as_lmmb_")
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr_as_lmmb.cov_")
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covariate_test.val, pheno_test.val,"g.",covariate_test.val, predicted,"r.")
                    pylab.xlim([-1, 10])
                    pylab.errorbar(covariate_test.val, predicted,yerr,linestyle='None')
                    pylab.suptitle("test on test: test X to true target (green) and prediction (red)")
                    pylab.show()
                    ## Plot y and predicted y (test on train)
                    #pylab.plot(pheno_test.val,predicted_pheno.val,".")
                    #pylab.suptitle(name+": test on test: true target to prediction")
                    #pylab.show()

                self.compare_files(predicted_pheno,"lr_as_lmmb_")
                self.compare_files(covar2,"lr_as_lmmb.cov_")

    def test_lr2(self):
        do_plot = False

        import pylab
        logging.info("TestLmmTrain test_lr2")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        #make covar just numbers 0,1,...
        covar = self.covariate_whole.read()
        covar.val = np.array([[float(num)] for num in range(covar.iid_count)])
        covariate_train = covar[train_idx,:].read()
        covariate_test = covar[test_idx,:].read()
        K0_whole_test = KernelIdentity(covar.iid,covariate_test.iid)

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
        for name,K0_train,K0_whole_test in [("Identity Kernel",
                                            KernelIdentity(self.snpreader_whole.iid[train_idx]),
                                            KernelIdentity(self.snpreader_whole.iid,test=self.snpreader_whole.iid[test_idx])),
                                      #!!!later("sid_count=0", self.snpreader_whole[train_idx,[]],self.snpreader_whole[test_idx,[]])
                                      ]:
            logging.info(name)
            first_name = first_name or name
            #Learn model, save, load
            fastlmmx = FastLMM(GB_goal=2).fit(K0_train=K0_train, X=covariate_train, y=pheno_train)
                
                
            filename = self.tempout_dir + "/model_lr2.flm.p"
            fastlmm = fastlmmx

            do_test_on_train = True
            if do_test_on_train:
                #Predict with model (test on train)
                predicted_pheno, covar = fastlmm.predict(K0_whole_test=K0_train, X=covariate_train,count_A1=False) #test on train
                output_file = self.file_name("lr2a_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr2a.cov_"+name)
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
            predicted_pheno, covar  = fastlmm.predict(K0_whole_test=K0_whole_test, X=covariate_test,count_A1=False) #test on train
            output_file = self.file_name("lr2b_"+name)
            Dat.write(output_file,predicted_pheno)
            covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
            output_file = self.file_name("lr2b.cov_"+name)
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


    def test_str2(self):
        logging.info("TestLmmTrain test_str2")


        #Standardize train and test together
        whole_kernel = self.snpreader_whole.read_kernel(Unit())

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids
        covariate_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]

        K0_train_filename = self.tempout_dir + "/model_str2.kernel.npz"
        pstutil.create_directory_if_necessary(K0_train_filename)
        from pysnptools.kernelreader import KernelNpz
        KernelNpz.write(K0_train_filename,whole_kernel[train_idx].read(order='A',view_ok=True))

        fastlmm1 = FastLMM(GB_goal=2).fit(K0_train=K0_train_filename, X=covariate_train, y=pheno_train)
        filename = self.tempout_dir + "/model_str2.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1
                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covariate_test = self.covariate_whole[test_idx,:]

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=whole_kernel[:,test_idx].read(order='A',view_ok=True), X=covariate_test,count_A1=False)

        output_file = self.file_name("str2")
        Dat.write(output_file,predicted_pheno)

        #pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]
        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"str2")

    #Creating multiple tests so that will run faster when on cluster.
    def test_fasttwoK(self):
        logging.info("TestLmmTrain test_fasttwoK")
        self._fasttwoK(None,None)

    def test_fasttwoK_force_low_rank(self):
        logging.info("TestLmmTrain test_fasttwoK_force_low_rank")
        self._fasttwoK(True,None)

    def test_fasttwoK_GB2(self):
        logging.info("TestLmmTrain test_fasttwoK_GB2")
        self._fasttwoK(None,2)

    def test_fasttwoK_force_low_rank_GB2(self):
        logging.info("TestLmmTrain test_fasttwoK_force_low_rank_GB2")
        self._fasttwoK(True,2)

    def _fasttwoK(self,force_low_rank,GB_goal):

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        G1_train = SnpData(iid=G0_train.iid,sid=[item+"_1" for item in G0_train.sid],val=G0_train.read().val,pos=G0_train.pos,name="Different SNP names for {0}".format(G0_train))
        covariate_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]

        logging.info("force_low_rank = {0}".format(force_low_rank))
        fastlmm1 = FastLMM(force_low_rank=force_low_rank,GB_goal=GB_goal).fit(K0_train=G0_train, K1_train=G1_train, X=covariate_train, y=pheno_train, mixing=.1)

        filename = self.tempout_dir + "/model_fasttwoK.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1
                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        G1_test = SnpData(iid=G0_test.iid,sid=[item+'_1' for item in G0_test.sid],val=G0_test.read().val,pos=G0_test.pos,name="Different SNP names for {0}".format(G0_test))
        covariate_test = self.covariate_whole[test_idx,:]

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=G0_test, K1_whole_test=G1_test, X=covariate_test,count_A1=False)

        output_file = self.file_name("fasttwoK"+("_force_low" if force_low_rank else "")+("GB{0}".format(GB_goal) if GB_goal is not None else ""))
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one")

    def test_lowrank(self):
        logging.info("TestLmmTrain test_lowrank")

        snpreader = self.snpreader_whole[:,:100]

        train_idx = np.r_[10:snpreader.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = snpreader[train_idx,:]
        G0_test = snpreader[test_idx,:]

        pheno_whole = self.pheno_whole.read()
        pheno_whole.val *= 100
        pheno_whole.val += 1000

        mean_low, covar_low =   FastLMM(force_low_rank=True,GB_goal=2).fit(K0_train=G0_train, y=pheno_whole[train_idx,:], X=self.covariate_whole[train_idx,:]). predict(K0_whole_test=G0_test,X=self.covariate_whole[test_idx,:],count_A1=False)
        mean_full, covar_full = FastLMM(force_full_rank=True,GB_goal=2).fit(K0_train=G0_train, y=pheno_whole[train_idx,:], X=self.covariate_whole[train_idx,:]).predict(K0_whole_test=G0_test,X=self.covariate_whole[test_idx,:],count_A1=False)

        np.testing.assert_allclose(mean_low.val, mean_full.val)
        np.testing.assert_allclose(covar_low.val,covar_full.val)

        logging.info("finished with TestLmmTrain test_lowrank")

    def test_twoK(self):
        logging.info("TestLmmTrain test_twoK")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        covariate_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]

        fastlmm1 = FastLMM(GB_goal=2).fit(K0_train=G0_train, K1_train=G0_train, X=covariate_train, y=pheno_train)
        filename = self.tempout_dir + "/model_one.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1

                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covariate_test = self.covariate_whole[test_idx,:]

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=G0_test, K1_whole_test=G0_test, X=covariate_test,count_A1=False)

        output_file = self.file_name("one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one")

    def test_lr(self):
        import matplotlib.pyplot as plt
        import pylab


        logging.info("TestLmmTrain test_lr")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        covariate_train3 = self.covariate_whole[train_idx,:].read()
        covariate_train3.val = np.array([[float(num)] for num in range(covariate_train3.iid_count)])
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        np.random.seed(0)
        pheno_train3.val = covariate_train3.val * 2.0 + 100 + np.random.normal(size=covariate_train3.val.shape) # y = 2*x+100+normal(0,1)

        ##Plot training x and y
        #pylab.plot(covariate_train3.val, pheno_train3.val,".")
        #pylab.show()

        for force_full_rank,force_low_rank in [(True,False),(False,True)]:
            #Learn model, save, load
            fastlmm3x = FastLMM(force_full_rank=force_full_rank,force_low_rank=force_low_rank,GB_goal=2).fit(K0_train=G0_train, X=covariate_train3, y=pheno_train3)
            filename = self.tempout_dir + "/model_lr.flm.p"
            pstutil.create_directory_if_necessary(filename)
            fastlmm3 = fastlmm3x


            #Predict with model (test on train)
            predicted_pheno, covar = fastlmm3.predict(K0_whole_test=G0_train, X=covariate_train3,count_A1=False) #test on train
            output_file = self.file_name("lr")
            Dat.write(output_file,predicted_pheno)

            ## Plot training x and y, and training x with predicted y
            #do_plot = True 
            #if do_plot:
            #    pylab.plot(covariate_train3.val, pheno_train3.val,covariate_train3.val,predicted_pheno.val,".")
            #    pylab.show()

            #    # Plot y and predicted y (test on train)
            #    pheno_actual = pheno_train3.val[:,0]
            #    pylab.plot(pheno_actual,predicted_pheno.val,".")
            #    pylab.show()


            self.compare_files(predicted_pheno,"lr")

    def test_lmm(self):
        do_plot = False
        iid_count = 500
        seed = 0


        import pylab
        logging.info("TestLmmTrain test_lmm")

        iid = [["cid{0}P{1}".format(iid_index,iid_index//250)]*2 for iid_index in range(iid_count)]
        train_idx = np.r_[10:iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids


        #Every person is 100% related to everyone in one of 5 families
        K0a = KernelData(iid=iid,val=np.empty([iid_count,iid_count]),name="related by distance")
        for iid_index0 in range(iid_count):
            for iid_index1 in range(iid_count):
                K0a.val[iid_index0,iid_index1] = 1 if iid_index0 % 5 == iid_index1 % 5 else 0
                if iid_index1 < iid_index0:
                    assert K0a.val[iid_index0,iid_index1] == K0a.val[iid_index1,iid_index0]

        #every person lives on a line from 0 to 1
        # They are related to every other person as a function of distance on the line
        np.random.seed(seed)
        home = np.random.random([iid_count])
        K0b = KernelData(iid=iid,val=np.empty([iid_count,iid_count]),name="related by distance")
        for iid_index in range(iid_count):
            K0b.val[iid_index,:] = 1 - np.abs(home-home[iid_index])**.1

        #make covar just numbers 0,1,...
        covar = SnpData(iid=iid,sid=["x"],val=np.array([[float(num)] for num in range(iid_count)]))
        covariate_train = covar[train_idx,:].read()
        covariate_test = covar[test_idx,:].read()

        for name, h2, K0 in [("clones", 1, K0a),("line_world",.75,K0b)]:

            sigma2x = 100
            varg = sigma2x * h2
            vare = sigma2x * (1-h2)

            #######################################################################
            #make pheno  # pheno = 2*covar+100+normal(0,1)*2.5+normal(0,K)*7.5
            #######################################################################
            #random.multivariate_normal is sensitive to mkl_num_thread, so we control it.
            if 'MKL_NUM_THREADS' in os.environ:
                mkl_num_thread = os.environ['MKL_NUM_THREADS']
            else:
                mkl_num_thread = None
            os.environ['MKL_NUM_THREADS'] = '1'
            np.random.seed(seed)
            p1 = covar.val * 2.0 + 100
            p2 = np.random.normal(size=covar.val.shape)*np.sqrt(vare)
            p3 = (np.random.multivariate_normal(np.zeros(iid_count),K0.val)*np.sqrt(varg)).reshape(-1,1)
            if mkl_num_thread is not None:
                os.environ['MKL_NUM_THREADS'] = mkl_num_thread
            else:
                del os.environ['MKL_NUM_THREADS']
            pheno = SnpData(iid=iid,sid=["pheno0"],val= p1 + p2 + p3)

            pheno_train = pheno[train_idx,:].read()
            pheno_test = pheno[test_idx,:].read()

            if do_plot:
                #Plot training x and y, testing x and y
                pylab.plot(covariate_train.val, pheno_train.val,".",covariate_test.val, pheno_test.val,".")
                pylab.suptitle(name + ": Plot training x and y, testing x and y")
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
                pylab.suptitle(name + ": real linear regression: actual to prediction")
                pylab.show()

            for factor in [1,100,.02]:
                K0 = K0.read()
                K0.val *= factor

                K0_train = K0[train_idx]
                K0_whole_test = K0[:,test_idx]

                #Learn model, save, load
                fastlmmx = FastLMM(GB_goal=2).fit(K0_train=K0_train, X=covariate_train, y=pheno_train)
                v2 = np.var(p2)
                v3 = np.var(p3)
                logging.debug("Original h2 of {0}. Generated h2 of {1}. Learned h2 of {2}".format(h2, v3/(v2+v3), fastlmmx.h2raw))
                
                
                filename = self.tempout_dir + "/model_lmm.flm.p"
                pstutil.create_directory_if_necessary(filename)
                fastlmm = fastlmmx

                do_test_on_train = True
                if do_test_on_train:
                    #Predict with model (test on train)
                    predicted_pheno, covar_pheno = fastlmm.predict(K0_whole_test=K0_train, X=covariate_train,count_A1=False) #test on train
                    output_file = self.file_name("lmma_"+name)
                    Dat.write(output_file,predicted_pheno)
                    covar2 = SnpData(iid=covar_pheno.row,sid=covar_pheno.col[:,1],val=covar_pheno.val) #kludge to write kernel to text format
                    output_file = self.file_name("lmma.cov_"+name)
                    Dat.write(output_file,covar2)

                    yerr = np.sqrt(np.diag(covar_pheno.val))
                    predicted = predicted_pheno.val
                    if do_plot:
                        pylab.plot(covariate_train.val, pheno_train.val,"g.",covariate_train.val, predicted,"r.")
                        pylab.xlim([0, 50])
                        pylab.ylim([100, 200])
                        pylab.errorbar(covariate_train.val, predicted,yerr,linestyle='None')
                        pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                        pylab.show()

                    self.compare_files(predicted_pheno,"lmma_"+name)
                    self.compare_files(covar2,"lmma.cov_"+name)

                    predicted_pheno0, covar_pheno0 = fastlmm.predict(K0_whole_test=K0_train[:,0], X=covariate_train[0,:],count_A1=False) #test on train #0
                    assert np.abs(predicted_pheno0.val[0,0] - predicted_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"
                    assert np.abs(covar_pheno0.val[0,0] - covar_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"


                #Predict with model (test on test)
                predicted_phenoB, covar_phenoB  = fastlmm.predict(K0_whole_test=K0_whole_test, X=covariate_test,count_A1=False) #test on test
                output_file = self.file_name("lmmb_"+name)
                Dat.write(output_file,predicted_phenoB)
                covar2 = SnpData(iid=covar_phenoB.row,sid=covar_phenoB.col[:,1],val=covar_phenoB.val) #kludge to write kernel to text format
                output_file = self.file_name("lmmb.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar_phenoB.val))
                predicted = predicted_phenoB.val
                if do_plot:
                    pylab.plot(covariate_test.val, pheno_test.val,"g.",covariate_test.val, predicted,"r.")
                    pylab.xlim([-1, 10])
                    pylab.errorbar(covariate_test.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_phenoB,"lmmb_"+name)
                self.compare_files(covar2,"lmmb.cov_"+name)

                predicted_phenoB0, covar_phenoB0  = fastlmm.predict(K0_whole_test=K0_whole_test[:,0], X=covariate_test[0,:],count_A1=False) #test on a single test case
                assert np.abs(predicted_phenoB0.val[0,0] - predicted_phenoB.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"
                assert np.abs(covar_phenoB0.val[0,0] - covar_phenoB.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"

                #Predict with model test on some train and some test
                some_idx = list(range(covar.iid_count))
                some_idx.remove(train_idx[0])
                some_idx.remove(test_idx[0])
                covariate_some = covar[some_idx,:]
                K0_whole_some = K0[:,some_idx]
                predicted_phenoC, covar_phenoC  = fastlmm.predict(K0_whole_test=K0_whole_some, X=covariate_some,count_A1=False)
                for idxC, iidC in enumerate(predicted_phenoC.iid):
                    meanC = predicted_phenoC.val[idxC]
                    varC = covar_phenoC.val[idxC,idxC]
                    if iidC in predicted_pheno.iid:
                        predicted_pheno_ref = predicted_pheno
                        covar_pheno_ref = covar_pheno
                    else:
                        assert iidC in predicted_phenoB.iid
                        predicted_pheno_ref = predicted_phenoB
                        covar_pheno_ref = covar_phenoB
                    idx_ref = predicted_pheno_ref.iid_to_index([iidC])[0]
                    mean_ref = predicted_pheno_ref.val[idx_ref]
                    var_ref = covar_pheno_ref.val[idx_ref,idx_ref]
                    assert np.abs(meanC - mean_ref) < 1e-6
                    assert np.abs(varC - var_ref) < 1e-6


    def test_snps(self):
        logging.info("TestLmmTrain test_snps")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        # Show it using the snps
        G0_train = self.snpreader_whole[train_idx,:]
        covariate_train3 = self.covariate_whole[train_idx,:].read()
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        pheno_train3.val = G0_train[:,0:1].read().val*2

        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val[:,0],".")
        #pylab.show()

        #Learn model, save, load
        fastlmm3x = FastLMM(GB_goal=2).fit(K0_train=G0_train, X=covariate_train3, y=pheno_train3)
        filename = self.tempout_dir + "/model_snps.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm3 = fastlmm3x


        #Predict with model (test on train)
        predicted_pheno, covar = fastlmm3.predict(K0_whole_test=G0_train, X=covariate_train3,count_A1=False) #test on train
        output_file = self.file_name("snps")
        Dat.write(output_file,predicted_pheno)

        ### Plot training x and y, and training x with predicted y
        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val,".",G0_train[:,0:1].read().val[:,0],predicted_pheno.val,".")
        #pylab.show()

        ### Plot y and predicted y (test on train)
        #pheno_actual = pheno_train3.val[:,0]
        #pylab.plot(pheno_actual,predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"snps")

    def test_kernel(self):
        logging.info("TestLmmTrain test_kernel")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        # Show it using the snps
        K0_train = self.snpreader_whole[train_idx,:].read_kernel(Unit())
        covariate_train3 = self.covariate_whole[train_idx,:].read()
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        pheno_train3.val = self.snpreader_whole[train_idx,0:1].read().val*2
        assert np.array_equal(K0_train.iid,covariate_train3.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"
        assert np.array_equal(K0_train.iid,pheno_train3.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"

        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val[:,0],".")
        #pylab.show()

        #Learn model, save, load
        fastlmm3x = FastLMM(GB_goal=2).fit(K0_train=K0_train, X=covariate_train3, y=pheno_train3)
        filename = self.tempout_dir + "/model_snps.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm3 = fastlmm3x


        #Predict with model (test on train)
        predicted_pheno, covar = fastlmm3.predict(K0_whole_test=K0_train, X=covariate_train3,count_A1=False) #test on train
        output_file = self.file_name("kernel")
        Dat.write(output_file,predicted_pheno)

        #### Plot training x and y, and training x with predicted y
        #pylab.plot(self.snpreader_whole[train_idx,0:1].read().val[:,0], pheno_train3.val,".",self.snpreader_whole[train_idx,0:1].read().val[:,0],predicted_pheno.val,".")
        #pylab.show()

        #### Plot y and predicted y (test on train)
        #pheno_actual = pheno_train3.val[:,0]
        #pylab.plot(pheno_actual,predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"snps") #"kernel" and "snps" test cases should give the same results

    def test_kernel_one(self):
        logging.info("TestLmmTrain test_kernel_one")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        K0_train = SnpKernel(self.snpreader_whole[train_idx,:],standardizer=Unit())
        covariate_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]
        assert np.array_equal(K0_train.iid,covariate_train.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"
        assert np.array_equal(K0_train.iid,pheno_train.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"

        fastlmm1 = FastLMM(GB_goal=2).fit(K0_train=K0_train, X=covariate_train, y=pheno_train)
        filename = self.tempout_dir + "/model_kernel_one.flm.p"
        pstutil.create_directory_if_necessary(filename)
        fastlmm2 = fastlmm1

                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covariate_test = self.covariate_whole[test_idx,:]

        predicted_pheno, covar = fastlmm2.predict(K0_whole_test=G0_test, X=covariate_test,count_A1=False)

        output_file = self.file_name("kernel_one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one") #Expect same results as SNPs "one"

    def compare_files(self,answer,ref_base):
        reffile = TestFeatureSelection.reference_file("fastlmm/"+ref_base+".dat")
        reference=Dat(reffile).read()
        assert np.array_equal(answer.col,reference.col), "sid differs. File '{0}'".format(reffile)
        assert np.array_equal(answer.row,reference.row), "iid differs. File '{0}'".format(reffile)
        for iid_index in range(reference.row_count):
            for sid_index in range(reference.col_count):
                a_v = answer.val[iid_index,sid_index]
                r_v = reference.val[iid_index,sid_index]
                assert abs(a_v - r_v) < 1e-4 or abs(a_v - r_v)/abs(r_v) < 1e5, "Value at {0},{1} differs too much from file '{2}'".format(iid_index,sid_index,reffile)

    def test_r(self):
        from fastlmm.inference.fastlmm_predictor import FastLMM
        from pysnptools.snpreader.snpdata import SnpData
        np.random.seed(0)        

        fastlmm = FastLMM(GB_goal=2)
        iid_train = [[str(x), str(x)] for x in range(10)]
        iid_test = [[str(x), str(x)] for x in range(10, 20)]
        sid = ['a{0}'.format(str(x)) for x in range(20)]
        val_train = (np.random.random((10, 20)) > 0.5).astype(np.float)
        val_test = (np.random.random((10, 20)) > 0.5).astype(np.float)

        y = SnpData(iid=iid_train, sid=['label'],val=np.random.random((10, 1)))
        K0_train = SnpData(iid=iid_train,sid=sid,val=val_train)
        K0_test  = SnpData(iid=iid_test,sid=sid,val=val_test)
        fastlmm.fit(X=None, y=y, K0_train=K0_train)
        predicted_pheno, covar = fastlmm.predict(X=None, K0_whole_test=K0_test)

        output_file = self.file_name("test_r")
        Dat.write(output_file,predicted_pheno)
        self.compare_files(predicted_pheno,"test_r")



    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testmod(sys.modules['fastlmm.inference.fastlmm_predictor'])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFastLMM)
    return unittest.TestSuite([suite1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # this import is needed for the runner
    from fastlmm.inference.tests.test_fastlmm_predictor import TestFastLMM
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
        #runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5,just_one_process=True)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print(runner.run(distributable_test))


    logging.info("done with testing")
