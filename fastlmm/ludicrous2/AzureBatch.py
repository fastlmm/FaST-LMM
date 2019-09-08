import logging
import datetime
import os
import time
from collections import Counter
import unittest
import shutil
import tempfile
import numpy as np
import math
from pysnptools.util.mapreduce1 import  map_reduce
from pysnptools.util.mapreduce1.mapreduce import _identity, _dyn_vars
import pysnptools.util as pstutil

try:
    import dill as pickle
except:
    logging.warning("Can't import dill, so won't be able to clusterize lambda expressions. If you try, you'll get this error 'Can't pickle <type 'function'>: attribute lookup __builtin__.function failed'")
    import cPickle as pickle

try:
    import azure.batch.models as batchmodels
    import azure.storage.blob as azureblob
    azure_ok = True
except Exception, exception:
    logging.warning("Can't import azure, so won't be able to clusterize to azure")
    azure_ok = False

if azure_ok:
    import azurehelper as commonhelpers #!!! is this the best way to include the code from the Azure python sample's common.helper.py?
    import azure.batch.batch_service_client as batch 
    import azure.batch.batch_auth as batchauth 
    from onemil.blobxfer import run_command_string as blobxfer #https://pypi.io/project/blobxfer/


def deal(player_count, card_count):
    assert player_count <= card_count, "Some will get no resources"
    hand_count_list = [0]*player_count
    for card_index in xrange(card_count): #There is another way to do this that is linear in # of players instead of # of cards
        hand_count_list[card_index % player_count] += 1
    return hand_count_list
        
def deal_proportional(player_size_list, card_count):
    '''
    If player_size_list is the size of each player and card_count is the number of cards to deal,
    then this function will return a list of how many cards each player gets. The number of cards
    will be at least once and proportional to the player's size. For example, suppose the player_size_list
    is the size of each human chromosome as measured by base pair count. Then if card_count is the
    number of cluster nodes, then this will return a list of how many nodes should be assigned
    to each chromosome.
    '''
    player_size_list = np.array(player_size_list,dtype='float')
    #Give everyone one card
    assert len(player_size_list) <= card_count, "Expect there to be at least one card for every player"
    result = np.array([1]*len(player_size_list),dtype='float')
    card_count -= len(player_size_list)

    while card_count > 1:
        #Measure how out of proportion each player is
        score = result / player_size_list
        #Find the index of the smallest item
        index_of_most_out_of_proportion = np.argsort(score)[0]
        result[index_of_most_out_of_proportion] += 1
        card_count -= 1

    return list(np.array(result,dtype='int'))


class AzureBatch: # implements Irunner
    '''
    A class that implement the Irunner interface that map_reduce uses. It lets one run map_reduce work on
    an Azure batch account.

    **Constructor:**
        :Parameters: * **task_count** (*integer*) -- The number of tasks in the AzureBatch job.
                     * **pool_id_list** (*list of strings*) -- A list of names of the AzureBatch pool(s) in which to run.
                     * **with_one_pool** (*bool*) -- (default True) Run two-level map_reduceX as a single AzureBatch job, otherwise runs each
                                top-level value as its own job.
                     * **tree_scale_list** (*list of pairs*) -- (default None) If given, this a is list the same size as the pool_id_list.
                                for each pool_id, it gives a pair: A :class:`AzureP2P` and a string file name. When the job is running,
                                a monitor program will watch the AzureP2P and scale the number of nodes to no more than three times
                                the number of peer-to-peer copies of the file. As the tasks in the job finish, the monitor program will
                                scale the number of nodes down to the number of remaining tasks.
                     * **max_node_count_list** (*list of integers*) -- default None) If given, limits the maximum number of nodes in each pool.
                     * **mkl_num_threads** (*integer*) -- (default of number of processors on node) Limit the number of MKL threads used on a node.
                     * **update_python_path** ('once' [default], 'no', 'every_time') -- How often to transfer the code on the python_path to the nodes.
                     * **max_stderr_count** (*integer*) -- If some tasks fail, the maximum number of stderr files to display. Defaults to 5.
                     * **storage_credential** (*StorageCredential*) -- AzureBatch and AzureStorage credentials. If not given, created from ~/azurebatch/cred.txt.
                     * **storage_account_name** (*string*) Name of Azure storage account used to store run information. Defaults to first
                                   account listed in the cred.txt file.
                     * **show_log_diffs** (*bool*) (default True) -- If True, in-place log message will do a carriage return when the message changes.
    '''
    def __init__(self, task_count, pool_id_list, with_one_pool=True, tree_scale_list=None, max_node_count_list=None, mkl_num_threads=None, update_python_path="once", max_stderr_count=5,
                         storage_credential=None, storage_account_name=None, show_log_diffs=True,logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger() #!!! similar code elsewhere
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        logger.addHandler(logging_handler)

        self.taskcount = task_count
        self.with_one_pool = with_one_pool
        self.tree_scale_list = tree_scale_list
        self.mkl_num_threads = mkl_num_threads
        self.pool_id_list = pool_id_list
        self.update_python_path = update_python_path
        self.show_log_diffs = show_log_diffs

        self.container = "mapreduce3" #!!!make this an option
        self.utils_version = 3        #!!!make this an option
        self.pp_version = 3        #!!!make this an option
        self.data_version = 3        #!!!make this an option


        if storage_credential is None or isinstance(storage_credential,str):
            from onemil.azure_copy import StorageCredential
            storage_credential = StorageCredential(storage_credential)
        self.storage_credential = storage_credential
        self.storage_account_name = storage_account_name or self.storage_credential.storage_account_name_list[0]
        self.storage_key = storage_credential._account_name_to_key[self.storage_account_name]
        self.batch_client = storage_credential.batch_client()

        from onemil.monitor import Real
        self.world = Real(pool_id_list,tree_scale_list,max_node_count_list,self.batch_client,max_stderr_count=max_stderr_count)


        choices = ['once','every_time','no']
        assert update_python_path in choices, "Expect update_python_path to be {0}".format(",".join(["'{0}'".format(item) for item in choices]))
        if update_python_path == 'once':
            self._update_python_path_function()

    def _update_python_path_function(self):
        localpythonpath = os.environ.get("PYTHONPATH") #!!should it be able to work without pythonpath being set (e.g. if there was just one file)? Also, is None really the return or is it an exception.

        if localpythonpath == None: raise Exception("Expect local machine to have 'pythonpath' set")
        for i, localpathpart in enumerate(localpythonpath.split(';')):
            logging.info("Updating code on pythonpath as needed: {0}".format(localpathpart))
            blobxfer(r"blobxfer.py --skipskip --delete --storageaccountkey {0} --upload {1} {2}-pp-v{3}-{4} .".format(
                            self.storage_key,               #0
                            self.storage_account_name,      #1
                            self.container,                 #2
                            self.pp_version,                #3
                            i,                              #4
                            ),
                        wd=localpathpart)

    def _setup_job(self, distributable_list, pool_id, name, log_writer=None):
        '''
        This is the main method for submitting to AzureBatch.
        '''

        job_id = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")  + "-" + name.replace("_","-").replace("/","-").replace(".","-").replace("+","-").replace("(","").replace(")","")
        job_id_etc_list = []

        if True: # Pickle the things-to-run - put them in a local directory under the current directory called "runs/[jobid]" where the jobid is based on the date.
            if log_writer is not None: log_writer("{0}: Pickle the thing to run".format(name))
            run_dir_rel = os.path.join("runs",job_id)
            pstutil.create_directory_if_necessary(run_dir_rel, isfile=False)
            for index, distributable in enumerate(distributable_list):
                distributablep_filename = os.path.join(run_dir_rel, "distributable{0}.p".format(index))
                with open(distributablep_filename, mode='wb') as f:
                    pickle.dump(distributable, f, pickle.HIGHEST_PROTOCOL)

        if True: # Copy (update) any (small) input files to the blob
            if log_writer is not None: log_writer("{0}: Upload small input files".format(name))
            data_blob_fn = "{0}-data-v{1}".format(self.container,self.data_version)
            inputOutputCopier = AzureBatchCopier(data_blob_fn, self.storage_key, self.storage_account_name)
            script_list = ["",""] #These will be scripts for copying to and from AzureStorage and the cluster nodes.
            inputOutputCopier2 = AzureBatchCopierNodeLocal(data_blob_fn, self.container, self.data_version, self.storage_key, self.storage_account_name,
                                 script_list)
            for index, distributable in enumerate(distributable_list):
                inputOutputCopier.input(distributable)
                inputOutputCopier2.input(distributable)
                inputOutputCopier2.output(distributable)
                output_blobfn = "{0}/output{1}".format(run_dir_rel.replace("\\","/"),index) #The name of the directory of return values in Azure Storage.
                job_id_etc_list.append((job_id, inputOutputCopier, output_blobfn, run_dir_rel))

        if True: # Create the jobprep program -- sets the python path and downloads the pythonpath code. Also create node-local folder for return values.
            if log_writer is not None: log_writer("{0}: Create jobprep.bat script".format(name))
            localpythonpath = os.environ.get("PYTHONPATH") #!!should it be able to work without pythonpath being set (e.g. if there was just one file)? Also, is None really the return or is it an exception.
            jobprep_filename = os.path.join(run_dir_rel, "jobprep.bat")
            # It only copies down files that are needed, but with some probability (about 1 in 50, say) fails, so we repeat three times.
            with open(jobprep_filename, mode='w') as f2:
                f2.write(r"""set
set path=%AZ_BATCH_APP_PACKAGE_ANACONDA2%\Anaconda2;%AZ_BATCH_APP_PACKAGE_ANACONDA2%\Anaconda2\scripts\;%path%
for /l %%t in (0,1,3) do FOR /L %%i IN (0,1,{7}) DO python.exe %AZ_BATCH_TASK_WORKING_DIR%\blobxfer.py --skipskip --delete --storageaccountkey {2} --download {3} {4}-pp-v{5}-%%i %AZ_BATCH_NODE_SHARED_DIR%\{4}\pp\v{5}\%%i --remoteresource .
{6}
mkdir %AZ_BATCH_TASK_WORKING_DIR%\..\..\output{8}
exit /b 0
                """
                .format(
                    None,                                   #0 - not used
                    None,                                   #1 - not used
                    self.storage_key,                       #2
                    self.storage_account_name,              #3
                    self.container,                         #4
                    self.pp_version,                        #5
                    script_list[0],                         #6
                    len(localpythonpath.split(';'))-1,      #7
                    index,                                  #8
                ))

        if True: #Split the taskcount roughly evenly among the distributables
            subtaskcount_list = deal(len(distributable_list),self.taskcount)

        if True: # Create the map.bat and reduce.bat programs to run.
            if log_writer is not None: log_writer("{0}: Create map.bat and reduce.bat script".format(name))
            pythonpath_string = "set pythonpath=" + ";".join(r"%AZ_BATCH_NODE_SHARED_DIR%\{0}\pp\v{1}\{2}".format(self.container,self.pp_version,i) for i in xrange(len(localpythonpath.split(';'))))
            for index in xrange(len(distributable_list)):
                subtaskcount = subtaskcount_list[index]
                output_blobfn = job_id_etc_list[index][2]
                for i, bat_filename in enumerate(["map{0}.bat".format(index),"reduce{0}.bat".format(index)]):
                    bat_filename = os.path.join(run_dir_rel, bat_filename)
                    with open(bat_filename, mode='w') as f1:
                        #note that it's getting distributable.py from site-packages and never from the pythonpath
                        f1.write(r"""set path=%AZ_BATCH_APP_PACKAGE_ANACONDA2%\Anaconda2;%AZ_BATCH_APP_PACKAGE_ANACONDA2%\Anaconda2\scripts\;%path%
mkdir %AZ_BATCH_TASK_WORKING_DIR%\..\..\output{14}
{6}cd %AZ_BATCH_TASK_WORKING_DIR%\..\..\output{14}
{6}FOR /L %%i IN (0,1,{11}) DO python.exe %AZ_BATCH_JOB_PREP_WORKING_DIR%\blobxfer.py --storageaccountkey {2} --download {3} {8}/{10} . --remoteresource %%i.{0}.p
cd %AZ_BATCH_NODE_SHARED_DIR%\{8}\data\v{9}
{13}
python.exe %AZ_BATCH_APP_PACKAGE_ANACONDA2%\Anaconda2\Lib\site-packages\fastlmm\util\distributable.py %AZ_BATCH_JOB_PREP_WORKING_DIR%\distributable{14}.p LocalInParts(%1,{0},result_file=r\"{4}/result.p\",mkl_num_threads={1},temp_dir=r\"{4}\")
IF %ERRORLEVEL% NEQ 0 (EXIT /B %ERRORLEVEL%)
{6}{7}
cd %AZ_BATCH_TASK_WORKING_DIR%\..\..\output{14}
{5}for /l %%t in (0,1,3) do python.exe %AZ_BATCH_JOB_PREP_WORKING_DIR%\blobxfer.py --storageaccountkey {2} --upload {3} {8} %1.{0}.p --remoteresource {10}/%1.{0}.p
{6}for /l %%t in (0,1,3) do python.exe %AZ_BATCH_JOB_PREP_WORKING_DIR%\blobxfer.py --storageaccountkey {2} --upload {3} {8} result.p --remoteresource {10}/result.p
                        """
                        .format(
                            subtaskcount,                           #0
                            self.mkl_num_threads,                   #1
                            self.storage_key,                       #2
                            self.storage_account_name,              #3
                            "%AZ_BATCH_TASK_WORKING_DIR%/../../output{0}".format(index), #4
                            "" if i==0 else "@rem ",                #5
                            "" if i==1 else "@rem ",                #6
                            script_list[1],                         #7
                            self.container,                         #8
                            self.data_version,                      #9
                            output_blobfn,                          #10
                            subtaskcount-1,                         #11
                            self.pp_version,                        #12
                            pythonpath_string,                      #13
                            index,                                  #14
                        ))

        if True: # Upload the thing-to-run to a blob and the blobxfer program
            if log_writer is not None: log_writer("{0}: Upload the thing to run".format(name))
            block_blob_client = azureblob.BlockBlobService(account_name=self.storage_account_name,account_key=self.storage_key)
            block_blob_client.create_container(self.container, fail_on_exist=False)

            blobxfer_blobfn = "utils/v{}/blobxfer.py".format(self.utils_version)
            blobxfer_url   = commonhelpers.upload_blob_and_create_sas(block_blob_client, self.container, blobxfer_blobfn, os.path.join(os.path.dirname(__file__),"blobxfer.py"), datetime.datetime.utcnow() + datetime.timedelta(days=30))

            jobprep_blobfn = "{}/jobprep.bat".format(run_dir_rel.replace("\\","/"))
            jobprepbat_url = commonhelpers.upload_blob_and_create_sas(block_blob_client, self.container, jobprep_blobfn, os.path.join(run_dir_rel, "jobprep.bat"), datetime.datetime.utcnow() + datetime.timedelta(days=30))

            map_reduce_url_list = []
            for index in xrange(len(distributable_list)):
                distributablep_blobfn = "{0}/distributable{1}.p".format(run_dir_rel.replace("\\","/"),index)
                distributablep_filename = os.path.join(run_dir_rel, "distributable{0}.p".format(index))
                distributablep_url = commonhelpers.upload_blob_and_create_sas(block_blob_client, self.container, distributablep_blobfn, distributablep_filename, datetime.datetime.utcnow() + datetime.timedelta(days=30)) #!!!should there be an expiry?

                map_blobfn = "{0}/map{1}.bat".format(run_dir_rel.replace("\\","/"),index)
                map_url = commonhelpers.upload_blob_and_create_sas(block_blob_client, self.container, map_blobfn, os.path.join(run_dir_rel, "map{0}.bat".format(index)), datetime.datetime.utcnow() + datetime.timedelta(days=30))

                reduce_blobfn = "{0}/reduce{1}.bat".format(run_dir_rel.replace("\\","/"),index)
                reduce_url = commonhelpers.upload_blob_and_create_sas(block_blob_client, self.container, reduce_blobfn, os.path.join(run_dir_rel, "reduce{0}.bat".format(index)), datetime.datetime.utcnow() + datetime.timedelta(days=30))
                map_reduce_url_list.append((map_url,reduce_url,distributablep_url))

        if True: # Copy everything on PYTHONPATH to a blob
            if log_writer is not None: log_writer("{0}: Upload items on pythonpath as requested".format(name))
            if self.update_python_path == 'every_time':
                self._update_python_path_function()

        if True: # Create a job with a job prep task
            if log_writer is not None: log_writer("{0}: Create jobprep.bat".format(name))
            resource_files=[
                batchmodels.ResourceFile(blob_source=blobxfer_url, file_path="blobxfer.py"),
                batchmodels.ResourceFile(blob_source=jobprepbat_url, file_path="jobprep.bat")]
            for index in xrange(len(distributable_list)):
                _, _, distributablep_url = map_reduce_url_list[index]
                resource_files.append(batchmodels.ResourceFile(blob_source=distributablep_url, file_path="distributable{0}.p".format(index)))
            
            job_preparation_task = batchmodels.JobPreparationTask(
                    id="jobprep",
                    #run_elevated=True,
                    user_identity=batchmodels.UserIdentity(auto_user=batchmodels.AutoUserSpecification(elevation_level='admin')),
                    resource_files=resource_files,
                    command_line="jobprep.bat",
                    )

            job = batchmodels.JobAddParameter(
                id=job_id,
                job_preparation_task=job_preparation_task,
                pool_info=batch.models.PoolInformation(pool_id=pool_id),
                uses_task_dependencies=True,
                on_task_failure='performExitOptionsJobAction',
                )
            try:
                self.batch_client.job.add(job)
            except batchmodels.BatchErrorException, e:
                if e.inner_exception.values is not None:
                    raise Exception(e.inner_exception.values[-1].value)
                else:
                    raise Exception(e.inner_exception)

        if True: # Add regular tasks to the job
            if log_writer is not None: log_writer("{0}: Add tasks to job".format(name))
            task_factor = int(10**math.ceil(math.log(max(subtaskcount_list),10))) #When we have multiple distributables, this helps us number them e.g. 0,1,2,10,11,12,20,21,22
            task_list = []
            for index in xrange(len(distributable_list)):
                start = len(task_list)
                map_url, reduce_url, _ = map_reduce_url_list[index]
                subtaskcount = subtaskcount_list[index]
                for taskindex in xrange(subtaskcount):
                    map_task = batchmodels.TaskAddParameter(
                        id=index * task_factor + taskindex,
                        #run_elevated=True,
                        user_identity=batchmodels.UserIdentity(auto_user=batchmodels.AutoUserSpecification(elevation_level='admin')),
                        #!!! seems to exit without needing a failure exit_conditions = batchmodels.ExitConditions(default=batchmodels.ExitOptions(job_action='terminate')),
                        resource_files=[batchmodels.ResourceFile(blob_source=map_url, file_path="map{0}.bat".format(index))],
                        command_line=r"map{0}.bat {1}".format(index, taskindex),
                    )
                    task_list.append(map_task)
                end = len(task_list)-1
                reduce_task = batchmodels.TaskAddParameter(
                    id="reduce{0}".format(index),
                    #run_elevated=True,
                    user_identity=batchmodels.UserIdentity(auto_user=batchmodels.AutoUserSpecification(elevation_level='admin')),
                    resource_files=[batchmodels.ResourceFile(blob_source=reduce_url, file_path="reduce{0}.bat".format(index))],
                    command_line=r"reduce{0}.bat {1}".format(index, subtaskcount),
                    depends_on = batchmodels.TaskDependencies(task_id_ranges=[batchmodels.TaskIdRange(task_list[start].id,task_list[end].id)])
                    )
                task_list.append(reduce_task)

            try:
                for i in xrange(0,len(task_list),100): #The Python API only lets us add 100 at a time.
                    self.batch_client.task.add_collection(job_id, task_list[i:i+100])
            except Exception as exception:
                print exception
                raise exception
        return job_id_etc_list

    def azure_storage_only(self):
        '''
        For all the AzureP2P's listed in tree_scale_list, remove all peer copies from the directory. (This is a good thing to do, for example,
        if all the nodes are being deallocated.)
        '''
        if self.tree_scale_list is not None:
            for azurep2p,_ in self.tree_scale_list:
                azurep2p.azure_storage_only()

    def run(self, distributable):
        '''
        This is the main method expected by map_reduce. It runs the work, given as distributable, on an AzureBatch cluster as one job.
        '''
        logging.info("Azure run: '{0}'".format(str(distributable) or ""))

        JustCheckExists().input(distributable) #Check that input files exist

        #!!! the default name for a map_reduce is "map_reduce()" which is illegal here. Fix that.
        name = str(distributable) or ""
        job_id, inputOutputCopier, output_blobfn, run_dir_rel = self._setup_job([distributable],self.pool_id_list[0],name)[0]
        self.world.drive_job_list([job_id],show_log_diffs=self.show_log_diffs)
        result = self._process_result(distributable, inputOutputCopier,output_blobfn, run_dir_rel)
        logging.info("About to run azure_storage_only in run")
        self.azure_storage_only()
        return result

    def run_list(self, distributable_list,reducer,name):
        '''
        This is the main method expected by map_reduceX. It runs the work, given as distributable, on an AzureBatch cluster as a set of related jobs.
        '''
        if self.with_one_pool:
            result = self._run_list_one_pool(distributable_list,reducer,name)
        else:
            result = self._run_list_multi_pool(distributable_list,reducer,name)
        logging.info("About to run azure_storage_only in run_list")
        self.azure_storage_only()
        return result

    def _run_list_one_pool(self, distributable_list,reducer,name):
        logging.info("Azure one_pool_list: {0}".format(",".join((str(d) for d in distributable_list))))

        if not isinstance(name,str): #If the name is not a string, assume it is a lambda
            name = name(0)+"_etc"

        assert len(self.pool_id_list) == 1, "The length of the pool_id_list should be one"

        job_id_etc_list = self._setup_job(distributable_list,self.pool_id_list[0],name)
        job_id = job_id_etc_list[0][0]
        self.world.drive_job_list([job_id],show_log_diffs=self.show_log_diffs)

        result_list = []
        for index, (job_id, inputOutputCopier, output_blobfn, run_dir_rel) in enumerate(job_id_etc_list):
            result = self._process_result(distributable_list[index], inputOutputCopier,output_blobfn, run_dir_rel)
            result_list.append(result)

        top_level_result = reducer(result_list)

        return top_level_result
        
    def _run_list_multi_pool(self, distributable_list,reducer,name):
        from onemil.azure_copy import log_in_place

        logging.info("Azure multi-pool run_list: {0}".format(",".join((str(d) for d in distributable_list))))

        assert len(self.pool_id_list) >= len(distributable_list), "The length of the pool_id_list should be at least as long as the list of top-level work to do"

        name_list = []
        for index in xrange(len(distributable_list)):
            if not isinstance(name,str): #If the name is not a string, assume it is a lambda
                name_in = name(index)
            else:
                namein  = name+"-{0}".format(index)
            name_list.append(name_in)


        with log_in_place("creating job_id_etc", logging.INFO) as log_writer:            
            job_id_etc_list = []
            for index,(distributable,pool_id,name_in) in enumerate(izip(distributable_list,self.pool_id_list,name_list)):
                log_writer(name_in)
                job_id_etc = self._setup_job([distributable],pool_id,name_in)[0]
                job_id_etc_list.append(job_id_etc)
        job_id_list = [job_id_etc[0] for job_id_etc in job_id_etc_list]
        self.world.drive_job_list(job_id_list,show_log_diffs=self.show_log_diffs)

        #If there are multiple reduces working with the same big file (e.g. postsvd's working with G0_data.dat), remove all references to the p2p copies because we don't know which one was last
        if self.tree_scale_list is not None and len(self.tree_scale_list) == 1 and len(distributable_list) > 1:
            azurep2p,_ = self.tree_scale_list[0]
            azurep2p.azure_storage_only()
        
        result_list = []
        for index, (job_id, inputOutputCopier, output_blobfn, run_dir_rel) in enumerate(job_id_etc_list):
            result = self._process_result(distributable_list[index], inputOutputCopier,output_blobfn, run_dir_rel)
            #logging.info("Result from {0}: '{1}'".format(job_id,result))
            result_list.append(result)

        top_level_result = reducer(result_list)

        return top_level_result

    def _process_result(self, distributable, inputOutputCopier,output_blobfn, run_dir_rel):

        inputOutputCopier.output(distributable) # Copy (update) any output files from the blob

        blobxfer(r"blobxfer.py --storageaccountkey {0} --download {1} {2}/{3} . --remoteresource result.p".format(self.storage_key,self.storage_account_name,self.container,output_blobfn), wd=run_dir_rel)
        resultp_filename = os.path.join(run_dir_rel, "result.p")
        with open(resultp_filename, mode='rb') as f:
            result = pickle.load(f)

        return result

    

class AzureBatchCopier(object): #Implements ICopier
    '''
    Copies small input files from the monitor computer to AzureStorage. Copies the small output files from AzureStorage to the monitor computer.
    '''

    def __init__(self, blob_fn, storage_key, storage_account):
        self.blob_fn = blob_fn
        self.storage_key=storage_key
        self.storage_account=storage_account

    def input(self,item):
        if isinstance(item, str):
            assert not os.path.normpath(item).startswith('..'), "Input files for AzureBatch must be under the current working directory. This input file is not: '{0}'".format(item)
            itemnorm = "./"+os.path.normpath(item).replace("\\","/")
            blobxfer(r"blobxfer.py --skipskip --storageaccountkey {} --upload {} {} {}".format(self.storage_key,self.storage_account,self.blob_fn,itemnorm),wd=".")
        elif hasattr(item,"copyinputs"):
            item.copyinputs(self)
        # else -- do nothing

    def output(self,item):
        if isinstance(item, str):
            itemnorm = "./"+os.path.normpath(item).replace("\\","/")
            blobxfer(r"blobxfer.py --skipskip --storageaccountkey {} --download {} {} {} --remoteresource {}".format(self.storage_key,self.storage_account,self.blob_fn, ".", itemnorm), wd=".")
        elif hasattr(item,"copyoutputs"):
            item.copyoutputs(self)
        # else -- do nothing

class AzureBatchCopierNodeLocal(object): #Implements ICopier
    '''
    Creates scripts that ...
    Copies small input files from AzureStorage to the node. Copies the small output files from the node to AzureStorage.
    '''

    def __init__(self, datablob_fn, container, data_version, storage_key, storage_account, script_list):
        assert len(script_list) == 2, "expect script list to be a list of length two"
        script_list[0] = ""
        script_list[1] = ""
        self.datablob_fn = datablob_fn
        self.container = container
        self.data_version = data_version
        self.storage_key=storage_key
        self.storage_account=storage_account
        self.script_list = script_list
        self.node_folder = r"%AZ_BATCH_NODE_SHARED_DIR%\{0}\data\v{1}".format(self.container,self.data_version)
        self.script_list[0] += r"mkdir {0}{1}".format(self.node_folder,os.linesep)

    def input(self,item):
        if isinstance(item, str):
            itemnorm = "./"+os.path.normpath(item).replace("\\","/")
            self.script_list[0] += r"cd {0}{1}".format(self.node_folder,os.linesep)
            self.script_list[0] += r"python.exe %AZ_BATCH_TASK_WORKING_DIR%\blobxfer.py --storageaccountkey {} --download {} {} {} --remoteresource {}{}".format(self.storage_key,self.storage_account,self.datablob_fn, ".", itemnorm, os.linesep)
        elif hasattr(item,"copyinputs"):
            item.copyinputs(self)
        # else -- do nothing

    def output(self,item):
        if isinstance(item, str):
            itemnorm = "./"+os.path.normpath(item).replace("\\","/")
            self.script_list[1] += r"python.exe %AZ_BATCH_JOB_PREP_WORKING_DIR%\blobxfer.py --storageaccountkey {} --upload {} {} {} --remoteresource {}{}".format(self.storage_key,self.storage_account,self.datablob_fn, ".", itemnorm, os.linesep)
        elif hasattr(item,"copyoutputs"):
            item.copyoutputs(self)
        # else -- do nothing

