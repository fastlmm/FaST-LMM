import numpy as np
import numpy.random
import scipy as sp
import logging
import unittest
import os.path
import time
import sys
import doctest
from fastlmm.util.runner import Local, Hadoop, Hadoop2, HPC, LocalMultiProc, LocalInParts, LocalFromRanges, LocalMapper, LocalReducer
from fastlmm.util.distributabletest import DistributableTest


from fastlmm.util.distributed_map import d_map
from fastlmm.util.distributed_map import dummy


class TestDistributedMap(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        currentFolder = os.path.dirname(os.path.realpath(__file__))
        self.fn = currentFolder + "/../../tests/datasets/dummy.txt"
        self.args = [(a,self.fn) for a in range(10)]

    def test_local_single(self):
        """
        test leave one chromosome out iterator
        """

        # run on 4 core locally
        runner = Local(4)
        result = d_map(dummy, self.args, runner, input_files=[self.fn])
        expect = ['', 'A', 'AA', 'AAA', 'AAAB', 'AAABB', 'AAABBB', 'AAABBBC', 'AAABBBCC', 'AAABBBCCC']

        assert expect == result

    def test_local_multiproc(self):
        """
        test leave one chromosome out iterator
        """

        # run on 4 core locally
        runner = LocalMultiProc(4)
        result = d_map(dummy, self.args, runner, input_files=[self.fn])
        expect = ['', 'A', 'AA', 'AAA', 'AAAB', 'AAABB', 'AAABBB', 'AAABBBC', 'AAABBBCC', 'AAABBBCCC']

        assert expect == result


class A (object): #implements IDistributable

    def __init__(self,work_count,sub_work_count, name="A",sub_name="B",tempdirectory="A",define_work_sequence_range=True):
        self._work_count = work_count
        self.sub_work_count = sub_work_count
        self.name = name
        self.sub_name = sub_name
        self.__tempdirectory = tempdirectory
        self.define_work_sequence_range = define_work_sequence_range
        if define_work_sequence_range:
            self.work_sequence_range = self._work_sequence_range

    @property
    def work_count(self):
        return self._work_count

    def work_sequence(self):
        return self._work_sequence_range(0,self.work_count)

    def _work_sequence_range(self,start,stop):
        assert 0<= start and start<= stop and stop <= self.work_count
        for work_index in xrange(start,stop):
            if self.sub_work_count == 0:
                yield lambda work_index=work_index : "{0}[{1}]".format(self.name,work_index)  # the 'work_index=work_index' is need to get around a strangeness in Python
            else:
                yield A(self.sub_work_count,0,name="{0}[{1}]".format(self.sub_name,work_index),sub_name=None)

    def reduce(self, result_sequence):
        work_index = 0
        for result in result_sequence:
            if self.sub_work_count == 0:
                assert result == "{0}[{1}]".format(self.name,work_index)
            else:
                assert result == "{0}[{1}]".format(self.sub_name,work_index)
            work_index += 1
        assert work_index == self.work_count
        return self.name

    @property
    def tempdirectory(self):
        return self.__tempdirectory

    def __str__(self):
        return "{0}({1},{2},{3},{4},{5})".format(self.__class__.__name__, self._work_count,self.sub_work_count, self.name,self.sub_name,self.__tempdirectory,self.define_work_sequence_range)
    def __repr__(self):
        return self.__str__


class TestDistributable(unittest.TestCase):
    '''
    This is a class for testing the distributable classes. It shouldn't be confused with DistributableTest
    which is a class for distributing any testing. 
    '''

    def test_one(self):
        assert "A" == Local().run(A(1,0))
        assert "A" == Local().run(A(10,0))
        assert "A" == Local().run(A(3,10))
        assert "A" == LocalFromRanges([2,4]).run(A(10,0))
        assert "A" == LocalFromRanges([2,9,10]).run(A(3,0))
        assert "A" == LocalFromRanges([2,9,10]).run(A(3,10))

        np.random.seed(0)
        for i in xrange(1000):
            end = int(np.random.uniform(low=1,high=15))
            a_work_count = int(np.random.uniform(low=1,high=15))
            b_work_count = int(np.random.uniform(low=0,high=3))
            extra_steps = int(np.random.uniform(low=0,high=end-1))
            if end > 1:
                list = sorted(np.random.random_integers(low=1,high=end-1,size=extra_steps))
            else:
                list = []
            list.append(end)
            if i % 100 == 0: logging.info("random test case # {0}".format(i))
            assert "A" == LocalFromRanges(list).run(A(a_work_count,b_work_count))

    def test_local_mapper(self):
        self.localmapper(define_work_sequence_range=True)
        self.localmapper(define_work_sequence_range=False)

    def localmapper(self,define_work_sequence_range):
        dist = A(2,0,define_work_sequence_range=define_work_sequence_range)

        import cStringIO
        instream0 = cStringIO.StringIO("0\n1\n2\n3")
        middlestream = cStringIO.StringIO()
        runner0 = LocalMapper(4,None,1,instream=instream0,outstream=middlestream)
        runner0.run(dist)
        instream1 = cStringIO.StringIO(middlestream.getvalue())
        runner1 = LocalReducer(4,None,1,instream=instream1)
        assert "A" == runner1.run(dist)


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestDistributable)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestDistributedMap)
    return unittest.TestSuite([suite1, suite2])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import fastlmm.util.testdistributable

    suites = unittest.TestSuite([
                                    fastlmm.util.testdistributable.getTestSuite(),
                                    ])
    suites.debug

    if False: #Standard test run
        r = unittest.TextTestRunner(failfast=False)
        r.run(suites)
    else: #Cluster test run
        runner = Local()
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done with testing")
