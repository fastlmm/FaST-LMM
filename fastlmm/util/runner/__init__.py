'''
A runner is executes a 'distributable' job. FastLmmSet is an example of a distributable job.
Local is an example of a runner. Local can execute FastLMMSet (or any other distributable job) on a local machine as a single process.
LocalMultiProc is another runner. It can execute FastLmmSet, etc on a multiple processes on a local machine.

All runners implement the IRunner interface (but Python doesn't check or enforce this)

interface IRunner
    def run(self, distributable):
        # returns the value computed by the distributable job


'''
import os
import dill
import fastlmm.util.util as util
from itertools import *

def work_sequence_to_result_sequence(work_sequence):
    '''
    Does all the work in a sequence of work items and returns a sequence of results.
    '''
    for work in work_sequence:
        if callable(work):
            result = work()
        else:
            result = run_all_in_memory(work)
        yield result

def run_all_in_memory(work):
    '''
    If callable, just calls
    If distributable, does all the work and then calls reduce on the results.
    '''
    if callable(work):
        return work()
    else:
        assert hasattr(work,"work_sequence"), "Your work item doesn't have a work_sequence. This can be caused by having 'map_reduce(nested=something,...)' where you should have 'map_reduce(mapper=something,...)' because the 'something' is not a nested 'map_reduce'. The exception can also be caused by giving a 'runner' on an inner, nested map_reduce."
        work_sequence = work.work_sequence()
        result_sequence = work_sequence_to_result_sequence(work_sequence)
        return work.reduce(result_sequence)


def create_task_file_name(workdirectory, taskindex, taskcount):
    '''
    Creates a file name to use for saving a set of results.
    '''
    task_file_name = os.path.join(workdirectory, "{0}.{1}.p".format(taskindex,taskcount))
    return task_file_name

    

def shape_to_desired_workcount(distributable, taskcount):
    workcount = distributable.work_count
    if workcount > taskcount:
        distributable = BatchUpWork(distributable, workcount, taskcount)
    else:
       if workcount < taskcount:
           if workcount == 0:
                distributable = MakeWork(distributable,taskcount)
           else:
                distributable = ExpandWork(distributable, workcount, taskcount)
    return distributable

def run_one_task(original_distributable, taskindex, taskcount, workdirectory):
    '''
    Does a fraction of the work (e.g. 1 of every 1000 work items) and then saves the results to single file.
    if taskindex == taskcount, does the reduce step
    '''

    if not 0 < taskcount: raise Exception("Expect taskcount to be positive")
    if not (0 <= taskindex and taskindex < taskcount+1) :raise Exception("Expect taskindex to be between 0 (inclusive) and taskcount (exclusive)")

    shaped_distributable = shape_to_desired_workcount(original_distributable, taskcount)

    if shaped_distributable.work_count != taskcount : raise Exception("Assert: expect workcount == taskcount")

    util.create_directory_if_necessary(workdirectory, isfile=False, robust=True)

    if (taskindex < taskcount):
        doMainWorkForOneIndex(shaped_distributable, taskcount, taskindex, workdirectory)
        return None
    else:
        result_sequence = work_sequence_from_disk(workdirectory, taskcount)
        return shaped_distributable.reduce(result_sequence)


def work_sequence_for_one_index(distributable, taskAndWorkcount, taskindex):
    if hasattr(distributable,"work_sequence_range"):
        is_first_and_only = True
        for work in distributable.work_sequence_range(taskindex, taskindex+1):
            assert is_first_and_only, "real assert"
            is_first_and_only = False
            yield work
    else:
        for workIndex, work in enumerate(distributable.work_sequence()):
            if workIndex == taskAndWorkcount : raise Exception("Expect len(work_sequence) to match work_count, but work_sequence was too long")
            if workIndex == taskindex :
                yield work
                workDone = True
                if workIndex != taskAndWorkcount-1  : #the work is done, so quit enumerating work (but don't quit early if you're the last workIndex because we want to double check that the work_sequence and work_count match up)
                    break
        if not workDone : raise Exception("Expect len(work_sequence) to match work_count, but work_sequence was too short")

def doMainWorkForOneIndex(distributable, taskAndWorkcount, taskindex, workdirectory):
    task_file_name = create_task_file_name(workdirectory, taskindex, taskAndWorkcount)
    workDone = False
    with open(task_file_name, mode='wb') as f:
        for work in work_sequence_for_one_index(distributable, taskAndWorkcount, taskindex):
            result = run_all_in_memory(work)
            dill.dump(result, f, dill.HIGHEST_PROTOCOL) # save a result to the temp results file

def work_sequence_from_disk(workdirectory, taskAndWorkcount):
    '''
    Reads all the result sets from the temporary files and returns a sequence results
    '''
    for taskindex in xrange(taskAndWorkcount):
        task_file_name = create_task_file_name(workdirectory, taskindex, taskAndWorkcount)
        with open(task_file_name, mode='rb') as f:
            try:
                result = dill.load(f)
            except Exception, detail:
                raise Exception("Error trying to unpickle '{0}'. {1}".format(task_file_name,detail))
        #if True:
        #    logging.debug("\tAbout to yield result {0} of {1}".format(taskindex,taskAndWorkcount))
        yield result


class BatchUpWork(object): # implements IDistributable
    '''
    A wrapper.
    Takes a distributable that has more work items that wanted and turns it into one with exactly the right number of work items.
    It does this by batching up work items, block-style style.
    '''
    def __init__(self, distributable, workcount, taskcount):
        self.sub_distributable = distributable
        self.sub_workcount = workcount
        self._workcount = taskcount

    @property
    def work_count(self):
        return self._workcount

    def work_sequence(self):
        return work_sequence_range(0,self._workcount)

    def work_sequence_range(self, start, stop):
        assert 0 <= start and start <= stop and stop <= self._workcount, "real assert"
        for workIndex in xrange(start, stop):
            yield lambda workIndex=workIndex : self.work(workIndex)
            
    def work(self,workIndex):
        result = [workIndex]
        start, stop = self.createSubWorkIndexList(workIndex)
        for sub_workIndex, sub_work in self.pull_sub_work(start, stop):
            sub_result = run_all_in_memory(sub_work)
            result.append((sub_workIndex,sub_result))
        return result

    def pull_sub_work(self, start, stop):
        if hasattr(self.sub_distributable,"work_sequence_range"):
            index = start
            for sub_work in self.sub_distributable.work_sequence_range(start,stop):
                yield index, sub_work
                index += 1
            assert index == stop, "real assert"
        else:
            sub_workIndexList = range(stop-1,start-1,-1)
            for sub_workIndex, sub_work in enumerate(self.sub_distributable.work_sequence()):
                if sub_workIndex == self.sub_workcount : raise Exception("Expect len(work_sequence) to match work_count")
                if sub_workIndex ==  sub_workIndexList[-1]:
                    yield sub_workIndex, sub_work
                    sub_workIndexList.pop()
                    if len(sub_workIndexList) == 0 and sub_workIndex != self.sub_workcount-1  : #there are no more work items for this task, so quit (but don't quit early if you're the last workIndex because we want to double check that the work_sequence and work_count match up)
                        break
            if len(sub_workIndexList) != 0 : raise Exception("no work was enumerated for workIndex={0}".format(sub_workIndexList[-1]))

    def reduce(self, result_sequence):
        sub_result_sequence = self.create_sub_result_sequence(result_sequence)
        result = self.sub_distributable.reduce(sub_result_sequence)
        return result

    def create_sub_result_sequence(self, result_sequence):
        for result in result_sequence:
            workIndex = result.pop(0)
            start, stop = self.createSubWorkIndexList(workIndex)
            if (stop-start) != len(result) : raise Exception("Assert: batched results not expected size")
            for sub_workIndex, pair in izip(xrange(start,stop), result):
                sub_workIndex_check, sub_result = pair
                if sub_workIndex != sub_workIndex_check : raise Exception("Assert: Unexpected workindex in batched result")
                yield sub_result

    @property
    def tempdirectory(self):
        return self.sub_distributable.tempdirectory

    #optional
    def __str__(self):
        return "{0}({1},{2},{3})".format(self.__class__.__name__,self.sub_distributable,self.sub_workcount,self._workcount)

    def createSubWorkIndexList(self, workindex):
        assert 0 <= workindex and workindex < self._workcount, "real assert"
        start = workindex * self.sub_workcount // self._workcount # assuming high prediction integer math
        stop = (workindex + 1) * self.sub_workcount // self._workcount # assuming high prediction integer math
        return start,stop

class ExpandWork(object): # implements IDistributable
    def __init__(self, distributable, workcount, taskcount):
        assert 1 <= workcount, "Expect workcount to be at least 1"
        assert workcount < taskcount, "ExpandWork expects workcount to be less than taskcount"
        self.sub_distributable = distributable
        self.sub_workcount = workcount
        self._workcount = taskcount
        self.floor = taskcount//workcount                 #all work gets this many tasks
        self.excess = taskcount - self.floor * workcount  #but there are this many extra tasks
        self.extra_index = (self.floor + 1) * self.excess #tasks before here, get one extra task

        if hasattr(distributable,"work_sequence_range"):
            self.work_sequence_range = self._work_sequence_range

    @property
    def work_count(self):
        return self._workcount

    def expand_to(self,sub_workIndex):
        if sub_workIndex < self.excess:
            return self.floor + 1
        else:
            return self.floor

    def index_to_sub_index(self,index):
        if index > self.extra_index:
            sub_index = self.excess + (index - self.extra_index) // self.floor
        else:
            sub_index = index // (self.floor + 1)
        return sub_index

    def sub_index_to_start(self,sub_index):
        if sub_index < self.excess:
            start = sub_index * (self.floor + 1)
        else:
            start = self.extra_index + (sub_index - self.excess) * self.floor
        return start

    def sub_sub_start_stop(self,sub_start,start,stop):
        full_start = self.sub_index_to_start(sub_start)
        full_end = self.sub_index_to_start(sub_start+1)
        if full_start < start:
            sub_sub_start = start - full_start
        else:
            sub_sub_start = 0

        if stop < full_end:
            sub_sub_stop = stop - full_start
        else:
            sub_sub_stop = full_end - full_start

        return sub_sub_start, sub_sub_stop
                    

    def sub_distributable_work_sequence_range(self,start,stop):
        if hasattr(self.sub_distributable,"work_sequence_range"):
            return self.sub_distributable.work_sequence_range(start,stop)
        else:
            return islice(self.sub_distributable.work_sequence(),start,stop)


    def _work_sequence_range(self,start,stop):

        sub_start = self.index_to_sub_index(start)
        sub_stop = self.index_to_sub_index(stop-1)+1
        sub_workIndex=sub_start
        workIndex=start
        for sub_work in self.sub_distributable_work_sequence_range(sub_start,sub_stop):
            expand_to = self.expand_to(sub_workIndex)
            sub_sub_start, sub_sub_stop = self.sub_sub_start_stop(sub_workIndex, start, stop)
            if not callable(sub_work): #i.e. a distributable job. If our expand_to is 3 and i's work_count is 9, then it is batched up 3x3. On the other hand, if its work_count is 2, then we ExpandWork from 2 to 3.
                shaped_distributable = shape_to_desired_workcount(sub_work, expand_to)
                for sub_sub_work in shaped_distributable.work_sequence_range(sub_sub_start, sub_sub_stop):
                    yield sub_sub_work
                    workIndex += 1
                #    expand_to -= 1
                #if expand_to != 0 :
                #    raise Exception("Assert: shaped_distributable doesn't have right number of work items")
                #assert sub_sub_index == sub_sub_end, "real assert"
            else:
                sub_sub_index = sub_sub_start
                if sub_sub_index == 0:
                    yield sub_work
                    workIndex += 1
                    sub_sub_index += 1
                for sub_sub_index2 in xrange(sub_sub_index, sub_sub_stop):
                    assert sub_sub_index2 == sub_sub_index, "real assert"
                    yield lambda: None
                    workIndex += 1
                    sub_sub_index += 1
                assert sub_sub_index == sub_sub_stop, "real assert"
            sub_workIndex+=1
        assert sub_workIndex == sub_stop, "real assert"
        assert workIndex == stop, "real assert"

    def work_sequence(self):
        return self._work_sequence_range(0,self.work_count)

    def reduce(self, result_sequence):
        sub_result_sequence = self.create_sub_result_sequence(result_sequence)
        result = self.sub_distributable.reduce(sub_result_sequence)
        return result

    def create_sub_result_sequence(self, result_sequence):
        # e.g. we had 11 work items, but we wanted 300 so each work item was expanded to 28 or 27 work items.
        # now we have 300 results that we must turn to 11 results.
        # If we had just padded with empty work we could just filter those out and we'd have 11. But we might
        # have gone down another level of map reduce.
        # We assume that the 300 items must arrive in order.
        sub_workIndex=0
        workIndex=0
        for sub_work in self.sub_distributable.work_sequence():
            expand_to = self.expand_to(sub_workIndex)
            if not callable(sub_work):
                shaped_distributable = shape_to_desired_workcount(sub_work, expand_to)
                sub_result = SubGen(result_sequence,expand_to)
                result = shaped_distributable.reduce(sub_result)
                if not hasattr(shaped_distributable,"suppress_yield"):
                    yield result
                workIndex += expand_to
            else:
                yield result_sequence.next()
                workIndex += 1
                for workindex in xrange(1, expand_to):
                    paddedResultToIgnore = result_sequence.next()
                    if paddedResultToIgnore != None: raise Exception("Assert: Expected 'None' result")
                    workIndex += 1
            sub_workIndex+=1
        if sub_workIndex != self.sub_workcount : raise Excpetion("Assert:  expect len(result_sequence) to match workcount")
        if workIndex != self._workcount : raise Exception("Assert: expect len(result_sequence) to match workcount")
        try:
            result_sequence.next() #should get an StopIternation here, which will be ignored.
            raise Exception("Assert: expect len(result_sequence) to match workcount. This can be caused by a 'reducer' that doesn't pull every input from its input sequence.")
        except StopIteration:
            pass #do nothing


    @property
    def tempdirectory(self):
        return self.sub_distributable.tempdirectory

    #optional
    def __str__(self):
        return "{0}({1},{2},{3})".format(self.__class__.__name__,self.sub_distributable,self.sub_workcount,self._workcount)


class MakeWork(object): # implements IDistributable
    def __init__(self, distributable, taskcount):
        self._workcount = taskcount
        self.sub_distributable = distributable

    suppress_yield = None #Just defining it is enough. The value doesn't matter.
    
    @property
    def work_count(self):
        return self._workcount

    def work_sequence_range(self,start,stop):
        for i in xrange(start,stop):
            yield lambda: None

    def work_sequence(self):
        return self._work_sequence_range(0,self.work_count)

    def reduce(self, result_sequence):
        for result in result_sequence: #We iterate through, just so everything is accounted for
            pass
        return "This return value should always be ignored"

    def __str__(self):
        return "{0}({1})".format(self.__class__.__name__,self._workcount)


class SubGen:
    def __init__(self, gen, count):
        self.gen = gen
        self.count = count

    def __iter__(self):
        return self
        
    def next(self):
        if self.count > 0:
            self.count -= 1
            return self.gen.next()
        else:
            raise StopIteration()


from .Local import *
from .LocalMultiProc import *
from .LocalMultiThread import *
from .LocalInParts import *
from .HPC import *
from .Hadoop import *
from .Hadoop2 import *
from .LocalMapper import *
from .LocalReducer import *
from .localfromranges import *
