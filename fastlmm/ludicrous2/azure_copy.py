import os
import sys
import shutil
import tempfile
import logging
import unittest
logging.basicConfig(level=logging.INFO)
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalMultiThread
import re
import datetime
import pytz
import pysnptools.util as pstutil
import time
import multiprocessing
import random
import cStringIO as StringIO
import random
from contextlib import contextmanager
from onemil.AzureBatch import AzureBatch
from pysnptools.util.mapreduce1.runner import Local
import azure.batch.models as batchmodels

try:
    import azure.storage.blob as azureblob
    azure_ok = True
except Exception, exception:
    logging.warning("Can't import azure, so won't be able to clusterize to azure")
    azure_ok = False

if azure_ok:
    import azure.batch.batch_service_client as batch 
    import azure.batch.batch_auth as batchauth 
    from onemil.blobxfer import run_command_string as blobxfer #https://pypi.io/project/blobxfer/

class StorageCredential(object):
    '''
    A class that managers credentials for Azure Storage and Azure Batch.

    **Constructor:**
        :Parameters: * **file_name** (*string*) -- The path of a text file containing credential and account information. Defaults to "~/azurebatch/cred.txt"

    The file looks like::

        https://flbatch2.westus.batch.azure.com    flbatch2    ABC...==
        /subscriptions/012...EF/resourceGroups/flstor2/providers/Microsoft.ClassicNetwork/virtualNetworks/vnet2
        flstor2 XYZ...==


    The first line has three tab-separated values:

    * the URL of your batch account. Found in the Azure Portal by clicking on the batch service.
    * the short name of the batch account
    * the batch account's primary access key. Found in the Azure Portal by again clicking on the batch service and then clicking on "Keys".

    The second line is the resource id of your virtual network.

    There is then one line for each of your (currently one) storage account(s). Each line has two tab-separated parts:

    * the short name of the storage account
    * the storage account's "key1". Found in the Azure Portal by clicking on the storage account and then clicking on "Access keys".
    '''
    def __init__(self, file_name = None):
        if file_name is None:
            file_name = os.path.expanduser("~")+"/azurebatch/cred.txt"
        with open(file_name) as fp:
            self.batch_service_url, self.batch_account, self.batch_key = fp.readline().strip().split('\t')
            self.vnet = fp.readline().strip()
            self._account_name_to_key = {}
            self.storage_account_name_list = []
            for line in fp:
                storage_account_name, storage_key = line.strip().split('\t')
                self.storage_account_name_list.append(storage_account_name)
                self._account_name_to_key[storage_account_name] = storage_key
        self._account_name_to_block_blob_service = {}

    def batch_client(self):
        credentials = batchauth.SharedKeyCredentials(self.batch_account, self.batch_key)
        result = batch.BatchServiceClient(credentials,base_url=self.batch_service_url)
        return result

    def get_pool(self, pool_id, vm_size="standard_d15_v2", node_count=0):
        '''
        Creates an Azure Batch pool.

        :param pool_id: The pool_id of the pool to create
        :type pool_id: string

        :param vm_size: the type of compute nodes in the pool. (Defaults to 'standard_d15_v2')
        :type vm_size: string

        :param node_count: The number of compute nodes to initially create in the pool. Defaults to 0.
        :type node_count: number

        :rtype: string
        :return: the pool_id of the create pool
        '''
        batch_client = self.batch_client()
        pool_list = list(batch_client.pool.list())

        for pool in pool_list:
            if pool.id == pool_id: #We have a pool with this pool_id, is it busy?
                node_list = list(batch_client.compute_node.list(pool.id))
                for node in node_list:
                    if node.running_tasks_count>0:
                        logging.info("pool '{0}' exists and is busy".format(pool_id))
                        break
                return pool.id

        logging.info("pool '{0}' does not exist and will be created".format(pool_id))
                    

        user_identity=batchmodels.UserIdentity(auto_user=batchmodels.AutoUserSpecification(elevation_level='admin'))
        start_task = batchmodels.StartTask(command_line="cmd /c %AZ_BATCH_APP_PACKAGE_STARTUP%\startup.bat",user_identity=user_identity,wait_for_success=True)
        new_pool = batchmodels.PoolAddParameter(id = pool_id, vm_size = vm_size,start_task=start_task)
        new_pool.target_dedicated = node_count
        new_pool.max_tasks_per_node = 1
        cloud_service_configuration = batchmodels.CloudServiceConfiguration(os_family=4)
        new_pool.cloud_service_configuration = cloud_service_configuration
        new_pool.application_package_references = [batchmodels.ApplicationPackageReference("anaconda2"),batchmodels.ApplicationPackageReference("startup")]

        #add the vnet        
        #The ARM resource identifier of the virtual network subnet which the compute nodes of the pool will join. The virtual 
        #network must be in the same region and subscription as the Azure Batch
        #account. This property can only be specified for pools created with a
        #cloudServiceConfiguration.
        #		value	u'Property subnetId should be of the form /subscriptions/{0}/resourceGroups/{1}/providers/{2}/virtualNetworks/{3}/subnets/{4}'	unicode
        new_pool.network_configuration = batchmodels.NetworkConfiguration(self.vnet+"/subnets/default")

        try:
            batch_client.pool.add(new_pool)
        except Exception, e:
            print e
        return name

    def _pool_id_to_pool(self, pool_id, pool_list=None):
        if pool_list is None:
            pool_list = list(self.batch_client().pool.list())
        for pool in pool_list:
            if pool.id == pool_id:
                return pool
        else:
            return None

    def pool_id_list_to_job_id_list(self, pool_id_list):
        from collections import defaultdict
        batch_client = self.batch_client()
        job_list = list(batch_client.job.list())

        #Make a list of all jobs every assigned to any of the pools of interest
        pool_id_to_job_list = defaultdict(list) 
        for job in job_list:
            if job.pool_info.pool_id not in pool_id_list:
                continue
            pool_id_to_job_list[job.pool_info.pool_id].append(job)

        #For each pool, find it latest job
        job_id_list = [] 
        for pool_id in pool_id_list:
            if pool_id not in pool_id_to_job_list:
                job_id_list.append(None)
            else:
                sorted_jobs = sorted(pool_id_to_job_list[pool_id], key=lambda job:job.creation_time)
                job_id_list.append(sorted_jobs[-1].id)
              
        return job_id_list



    def blobxfer_params(self,storage):
        root, storage_account_name, container = storage.split('/')
        assert root == '', "Expect path to start with '/'"
        storage_key = self._account_name_to_key[storage_account_name]
        return storage_account_name, storage_key, container

    def _split(self,azure_path):
        if azure_path.count('/') == 2:
            root, account_name, container_name = azure_path.split('/')
            sub_path = None
        else:
            root, account_name, container_name, sub_path = azure_path.split('/',3)
        assert root == '', "Expect path to start with '/'"
        if not account_name in self._account_name_to_block_blob_service:
            assert account_name in self._account_name_to_key, "Don't know key for account_name '{0}'".format(account_name)
            self._account_name_to_block_blob_service[account_name] = azureblob.BlockBlobService(account_name=account_name, account_key=self._account_name_to_key[account_name])
        block_blob_service = self._account_name_to_block_blob_service[account_name]
        self._create_container(block_blob_service, container_name)
        

        return block_blob_service, container_name, sub_path

    def _container_exists(self, block_blob_service, container_name):
        sleep_time = 1.0
        for try_index in xrange(50):
            try:
                return block_blob_service.exists(container_name=container_name)
            except Exception as e:
                if (   e.message.startswith("The specified resource name contains invalid characters") or
                       e.message.startswith("The specifed resource name contains invalid characters")): #Note there is a misspelling in the error message.
                    raise Exception("'{0}' contains invalid characters for an Azure container name".format(container_name))
                logging.info("_robust_exists exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        raise Exception("_robust_exists fails")

    def rmtree(self,folder,log_writer=None): #!!!this may need coode to work if called with just "/" or "/container"
        assert len(folder)>0 and folder[0]=='/', "Expect folder to start with '/'"
        if folder.strip('/').count('/')==1: #It is a container
            folder = '/'+folder.strip('/') #Remove any trailing '/'
            self._robust_delete_container(folder,log_writer=log_writer)
        else: #It is not a container:
            for block_blob_service, container_name, blob in self.list_blobs(folder,log_writer=log_writer):
                self._robust_delete_blob(block_blob_service, container_name, blob.name)

    def _robust_delete_container(self,azure_path,log_writer=None):
        block_blob_service, container_name, sub_path = self._split(azure_path)
        assert sub_path is None, "expect no sub_path"
        sleep_time = 1.0
        for try_index in xrange(50):
            try:
                block_blob_service.delete_container(container_name)
                return
            except Exception as e:
                if str(e).startswith("The specified container does not exist."):
                    return []
                logging.info("Exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        raise Exception("list_blobs fails")            


    def _robust_delete_blob(self, block_blob_service, container_name, blob_name):
        sleep_time = 1.0
        for try_index in xrange(50):
            try:
                block_blob_service.delete_blob(container_name, blob_name)
                return
            except Exception as e:
                if "The specified blob does not exist." in str(e):
                    logging.info("The specified blob does not exist, so continuing on")
                    return
                logging.info("robust delete_blob '{0}' exception #{1}, will sleep {2}: {3}".format(blob_name, try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        raise Exception("robust delete_blob fails")


    def get_blobetc_to_file_pointer(self,stream,blobetc,start=0,max_connections=None):
        block_blob_service,container_name,blob = blobetc
        sleep_time = 1.0
        is_ok = False
        for try_index in xrange(50):
            try:
                stream.seek(start)
                block_blob_service.get_blob_to_stream(container_name, blob.name, stream ,max_connections=max_connections)
                is_ok = True
                break
            except Exception as e:
                logging.info("In _robust_get_blob_to_stream, exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        if not is_ok:
            raise Exception("_robust_get_blob_to_stream fails")
        
    def create_blob_from_file_pointer(self,fp,start,stop,azure_path,max_connections):
        block_blob_service, container_name, sub_path = self._split(azure_path)
        sleep_time = 1.0
        is_ok = False
        for try_index in xrange(50):
            try:
                fp.seek(start)
                block_blob_service.create_blob_from_stream(container_name, sub_path, fp, count=stop-start,max_connections=max_connections) #http://azure.github.io/azure-storage-python/ref/azure.storage.blob.blockblobservice.html
                is_ok = True
                break
            except Exception as e:
                logging.info("Exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        if not is_ok:
            raise Exception("_robust_create_blob_from_stream fails")            



    def _create_container(self, block_blob_service, container_name):
        if not self._container_exists(block_blob_service, container_name):
            sleep_time = 1.0
            for try_index in xrange(50):
                try:
                    return block_blob_service.create_container(container_name)
                except Exception as e:
                    logging.info("In _robust_create_container, exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                    time.sleep(sleep_time)
                    sleep_time = min(60.0,sleep_time*1.1)
            raise Exception("_robust_create_container fails even after 50 tries")


    def list_blobs(self,azure_path,log_writer=None):
        block_blob_service, container_name, sub_path = self._split(azure_path)
        assert sub_path != '', "Azure path must have at least three parts: the storage account, the container, and a subpath"
        sleep_time = 1.0
        is_ok = False
        result = None
        for try_index in xrange(50):
            try:
                result = [] #list_blocks returns all blobs with this prefix. Need to be sure we have an exact match.
                if not block_blob_service.exists(container_name):
                    is_ok = True
                    break
                for blob in block_blob_service.list_blobs(container_name, sub_path):
                    result.append((block_blob_service, container_name, blob))
                    if log_writer is not None:
                        log_writer(blob.name)
                is_ok = True
                break
            except Exception as e:
                if str(e).startswith("The specified container does not exist."):
                    return []
                logging.info("Exception #{0}, will sleep {1}: {2}".format(try_index,sleep_time,e))
                time.sleep(sleep_time)
                sleep_time = min(60.0,sleep_time*1.1)
        if not is_ok:
            raise Exception("list_blobs fails")            
        return result

def azurep2p_cache_dict(container_prefix, storage_credential=None):
    """
    Create a cache_dict.

    :param container_prefix: A prefix to use when naming the the Azure Storage blob containers to use.
    :type container_prefix: string or None

    :param storage_credential: :class:`StorageCredential` describing the Azure Storage accounts to use.
        Defaults the the information in '~/azurestorage/~cred.txt'
    :type storage_credential: :class:`StorageCredential`

    :rtype: dictionary of :class:`AzureP2P`
    :return: A cache_dict
    """
    from onemil.AzureP2P import AzureP2P, ip_address_local

    storage_credential = storage_credential or StorageCredential()
    
    max_chrom = len(storage_credential.storage_account_name_list)-1
    cache_dict = {}
    for chrom in xrange(0,23):
        if chrom < len(storage_credential.storage_account_name_list):
            storage_account_name = storage_credential.storage_account_name_list[chrom]
        else:
            storage_account_name = storage_credential.storage_account_name_list[(chrom-1)%max_chrom+1]
        
        if chrom == 0: #We add an extra level to the first storage account so that removing it will not remove the container.
            extra = "/0"
        else:
            extra = "" 
        azure_path = "/{0}/{1}{2}{3}".format(storage_account_name, container_prefix, chrom, extra)
        storage = AzureP2P(azure_path,                                                            #The 'directory' on AzureStorage to use
                            local_lambda=ip_address_local,                                        #Function giving the local directory to use
                            storage_credential=storage_credential)

        if chrom == 0:
            storage_credential._split(azure_path) #This will preallocate the azure container
        cache_dict[chrom] = storage
    return cache_dict

class AzureShardContainer(object): #!!! could this gave a better name?
    def __init__(self, storage_credential=None, max_piece_size=20000000, process_count=None, max_connections=None):
        """
        Define a container on Azure Storage that stores files in shards.
        """
        if storage_credential is None or isinstance(storage_credential,str):
            storage_credential = StorageCredential(storage_credential)
        self._storage_credential = storage_credential
        self._max_piece_size = max_piece_size
        self._process_count = process_count
        self._max_connections = max_connections
        #self._same_time_count = same_time_count
        #self._has_run_once = False

    @property
    def process_count(self):
        if self._process_count is not None:
            return self._process_count
        return -(-multiprocessing.cpu_count() // 2)

    @property
    def max_connections(self):
        if self._max_connections is not None:
            return self._max_connections
        return -(-200//self.process_count)

    def upload(self, local_path, azure_path, do_sync_date=True, updater=None):
        """
        Upload a local file to the container.
        """
        assert os.path.exists(local_path), 'Expect local_path to exist: "{0}"'.format(local_path)
        #self._run_once()
        t0 = time.time()
        self.remove(azure_path)

        size = os.path.getsize(local_path)
        piece_count = self._get_piece_count(size)

        with _file_transfer_reporter("upload", size, updater=updater) as updater2:
            def mapper_closure(piece_index):
                t00 = time.time()
                start = size * piece_index // piece_count
                stop = size * (piece_index+1) // piece_count
                shard_size = stop - start
                blob_name = "{0}/{1}.{2}".format(azure_path,piece_index,piece_count)
                self._create_blob_from_stream(local_path,start,stop,blob_name)
                updater2(shard_size)
                if piece_index==piece_count-1:
                    self._create_blob_from_stream(local_path,stop,stop,"{0}/exists.txt".format(azure_path))
        
            map_reduce(
                range(piece_count),
                mapper=mapper_closure,
                runner=self._get_runner(),
                )


        if do_sync_date:
            self._sync_date(azure_path,local_path)

    def file_exists(self,azure_path):
        """
        Tell if a file exists in the container.
        """
        return not not self._find_blobs(azure_path + "/exists.txt",only_shards=False)

    def walk(self, folder):
        assert not self.file_exists(folder), "Cannot walk a file"
    
        for block_service,container_name,blob in self._storage_credential.list_blobs(folder + "/"): #The "/" is needed to stop it from matching the start of directory names
            if blob.name.endswith('/exists.txt'):
                name = "/" + block_service.account_name+"/"+container_name+"/"+blob.name
                result = os.path.split(name[len(folder)+1:])[0]
                yield result

    def remove(self,azure_path,log_writer=None):
        """
        Remove a file from the container.
        """
        for block_blob_service,container_name,blob in self._find_blobs(azure_path,only_shards=None): #!!! does this do the right thing if there are extra files? what if the names overlap e.g. /aa/bb/cc and /aa/bb/ccc
            if log_writer:
                log_writer(blob.name)
            block_blob_service.delete_blob(container_name,blob.name)

    def download(self,azure_path,local_path, do_sync_date=True,as_needed=True, updater=None): #!!!perhaps should download to a tmp file and then rename after everything works.
        """
        Download a file from the container.

                _file_transfer_reporter    : is a python context manager what is initialized with a size and that yields a updater method that can be called with a byte count as the download progresses.

        """
        #self._run_once()

        if as_needed and not self._download_needed_and_ready(local_path,azure_path):
            return

        t0 = time.time()
        blob_list = self._find_blobs_and_check(azure_path)

        piece_count = len(blob_list)
        start_stop_pairs = []
        start = 0
        for _,_,blob in blob_list:
            stop = start + blob.properties.content_length
            start_stop_pairs.append((start,stop))
            start = stop
        size = start_stop_pairs[-1][1] # The size is the last stop value

        pstutil.create_directory_if_necessary(local_path,isfile=True)
        local_path_temp = local_path+".temp" #!!! give it a unique name to ensure that it can't collide with a user's name.
        with open(local_path_temp,"wb") as fp: #Preallocate the local file to its full size
            if size > 0:
                fp.seek(size-1)
                fp.write("\0")

        with _file_transfer_reporter("download", size, updater=updater) as updater:
            def mapper_closure(piece_index):
                blobetc = blob_list[piece_index]
                start, stop = start_stop_pairs[piece_index]
                logging.debug("\tDownloading {0}/{4} {1}-{2} in '{3}'".format(piece_index, start,stop,local_path,piece_count))
                self._get_blobetc_to_stream(blobetc,local_path_temp,start,stop)
                updater(stop-start)
            
            name = "download." + os.path.basename(local_path) + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")  + str(random.random())
            map_reduce(
                    range(piece_count),
                    mapper = mapper_closure,
                    name = name,
                    runner=self._get_runner(),
                )

        if do_sync_date:
            self._sync_date(azure_path,local_path_temp)
        self._rename_no_matter_what(local_path_temp,local_path)


    def getmdate(self,azure_path):
        """
        Get date of file.
        """
        blob_list = self._find_blobs(azure_path,only_shards=True)
        _, _, blob0 = blob_list[0]
        remote_date = blob0.properties.last_modified
        return remote_date

    def _download_needed_and_ready(self,local_path,azure_path):
        blob_list = self._find_blobs(azure_path,only_shards=True)
        if not blob_list or not os.path.exists(local_path):
            return True
        local_date = datetime.datetime.utcfromtimestamp(os.path.getmtime(local_path)).replace(tzinfo=pytz.utc)
        _,_,blob0 = blob_list[0]
        remote_date = blob0.properties.last_modified
        if local_date >= remote_date:
            local_size = os.path.getsize(local_path)
            blob_size = sum((blob.properties.content_length for _,_,blob in blob_list))
            if local_size != blob_size:
                logging.info("The local date is later (or same), but file sizes to not match, so removing that local copy and getting a new copy. {0} (size={1}), {2} (size={3}))".format(local_path,local_size,azure_path,blob_size))
                os.remove(local_path)
                return True
            assert local_date == remote_date, "If dates are from same file, expect them to match exactly ({0},{1})".format(local_path,azure_path)
        return local_date < remote_date

    def rmtree(self,folder,log_writer=None):
        self._storage_credential.rmtree(folder,log_writer=log_writer)

    def _get_piece_count(self,size):
        if size <= self._max_piece_size:
            return 1
        return min(max(self.process_count,-(-size/self._max_piece_size)),size)

    def _find_blobs(self, azure_path, only_shards):
        blob_list = self._storage_credential.list_blobs(azure_path) #list_blocks returns all blobs with this prefix. Need to be sure we have an exact match.


        if len(blob_list) == 0:
            return []
        result = []
        for block_service,container_name,blob in blob_list:
            #  "hello/there.txt" matches "hello/there.txt/0"? Yes
            #  "hello/there.txt" matches "hello/there.txt"?   Yes
            #  "hello/there.txt" matches "hello/there.txt2"?  NO!
            name = "/" + block_service.account_name+"/"+container_name+"/"+blob.name
            if name == azure_path or name.startswith(azure_path + "/"):
                if not only_shards:
                    result.append((block_service,container_name,blob))
                else:
                    if re.match("[0-9]+.[0-9]+", os.path.basename(blob.name)):
                        result.append((block_service,container_name,blob))
        if only_shards:
            result = sorted(result,key=lambda b:int(os.path.splitext(os.path.basename(b[2].name))[0])) #So, "2.20" is sorted before "10.20"
        return result

    def _sync_date(self,azure_path,local_path):
        blobetc_list = self._find_blobs(azure_path,only_shards=True)
        assert len(blobetc_list)>0, "expected blobs not found ('{0}')".format(azure_path)
        _,_,blob0 = blobetc_list[0]
        remote_date = blob0.properties.last_modified
        timestamp = (remote_date - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
        with open(local_path, 'a'): #Touch the local file so that its mod time will be later than the remote time (see http://stackoverflow.com/questions/1158076/implement-touch-using-python)
            os.utime(local_path, (timestamp,timestamp))

    def _create_blob_from_stream(self,local_path,start,stop,blob_name):
        with open(os.path.normpath(local_path),"rb") as fp:
            self._storage_credential.create_blob_from_file_pointer(fp,start,stop,blob_name,max_connections=self.max_connections)
       
    def _get_blobetc_to_stream(self,blobetc,local_path,start,stop):
        with open(local_path,"r+b") as fp:
            self._storage_credential.get_blobetc_to_file_pointer(fp,blobetc,start=start,max_connections=self.max_connections)
        
    def _get_runner(self):
        if self.process_count == 1:
            return Local()
        else:
            return LocalMultiThread(self.process_count,just_one_process=False)

    def _find_blobs_and_check(self,azure_path):
        blob_list = self._find_blobs(azure_path,only_shards=True)
        piece_count = len(blob_list)
        assert piece_count > 0,  "Expect azure_path to exist: '{0}'".format(azure_path)
        for piece_index, (block_service,container_name,blob) in enumerate(blob_list):
            blob_name1 = "{0}/{1}.{2}".format(azure_path,piece_index,piece_count)
            blob_name2 = "/" + block_service.account_name+"/"+container_name+"/"+blob.name
            assert blob_name1 == blob_name2, "Expect blobs on azure to be of the form INDEX.COUNT with no missing files and no extra files. '{0}'".format(azure_path)
        return blob_list

    def _rename_no_matter_what(self,local_path_temp,local_path):
        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(local_path_temp,local_path)


 
class TestAzureShardContainer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.temp_parent = os.path.join(tempfile.gettempdir(), format(hash(os.times())))
        self.count = 0
        storage_credential = StorageCredential()
        self.container = AzureShardContainer(storage_credential)

        self.big_file_name = self.temp_parent+"/big.txt"
        pstutil.create_directory_if_necessary(self.big_file_name)
        self.big_size = int(500e6)
        with open(self.big_file_name,"wb") as fp: #Preallocate the local file to its full size
            if self.big_size > 0:
                fp.seek(self.big_size-1)
                fp.write("\0")

    def _temp_dir(self):
        self.count += 1
        return "{0}/{1}".format(self.temp_parent,self.count)

    @classmethod
    def tearDownClass(self):
        not os.path.exists(self.temp_parent) or shutil.rmtree(self.temp_parent)

    def test_one(self):
        logging.info("test_one")
        import filecmp

        azure_path = "/chrom1/azurecopypy/test/one/little.txt"
        local_path0 = self._temp_dir()+"/little0.txt"
        pstutil.create_directory_if_necessary(local_path0)
        shutil.copy(os.path.realpath(__file__),local_path0)
        local_path = self._temp_dir()+"/little.txt"

        if self.container.file_exists(azure_path):
            self.container.remove(azure_path)
        assert not self.container.file_exists(azure_path)
        self.container.upload(local_path0,azure_path)
        assert self.container.file_exists(azure_path)
        self.container.download(azure_path,local_path,as_needed=True)
        self.container.download(azure_path,local_path,as_needed=True) #Manually testing: see that it doesn't download again
        assert self.container.getmdate(azure_path) == datetime.datetime.utcfromtimestamp(os.path.getmtime(local_path)).replace(tzinfo=pytz.utc)
        assert filecmp.cmp(local_path0,local_path)
        self.container.remove(azure_path)
        self.container.rmtree('/'.join(azure_path.split('/')[:3]))
        os.remove(local_path)

    def zzztest_big_file_with_message(self): #!!!needs to be updated
        logging.info("test_big_file_with_message")

        azure_path = "big.txt"
        if self.container.file_exists(azure_path):
            self.container.remove(azure_path)
        assert not self.container.file_exists(azure_path)

        with _file_transfer_reporter("test_big_file_with_message",self.big_size) as updater:
            self.container.upload(self.big_file_name,azure_path,updater=updater)

        self.container.remove(azure_path)

    def zzztest_big_files_fileshare(self): #!!!needs to be updated
        logging.info("test_big_files_fileshare")
        from onemil.AzureP2P import AzureP2P, ip_address_local
        from onemil.file_cache import AzureStorage, PeerToPeer
        directory = AzureStorage(folder="testazureshardcontainer/fileshare2",local_lambda=ip_address_local,prefix="AzureDirectory",storage_account=self.storage_account, storage_key=self.storage_key)
        storage = PeerToPeer(directory=directory,local_lambda=ip_address_local)

        big_size = int(4e9)
        update_python_path = "once"
        double_it = True
        AzureBatch(task_count=1,pool_id="d14v2_300",update_python_path=update_python_path)

        while True:
            for n in [16,4,14,2,12,10,8]:
                storage.directory.rmtree()
                runner = AzureBatch(task_count=n*2 if double_it else n,pool_id="d14v2_300",update_python_path="no")
                #runner = Local()
                #runner = LocalMultiThread(n*2 if double_it else n)
                mbps_list = self._big_files_fileshare_internal(big_size,n,runner,storage,double_it)
                logging.info("{0}\t{1}".format(n,mbps_list))

    @staticmethod
    def _big_files_fileshare_internal(big_size,n,runner,storage,double_it):
        nn = n*2 if double_it else n
        def mapper(ii):
            i = ii//2 if double_it else i
            if ii%2==0 or not double_it:
                short_name = "big{0}.{1}.txt".format(big_size,i)
                if storage.file_exists(short_name):
                    storage.remove(short_name)
                with storage.open_write(short_name) as file_name:
                    with open(file_name,"wb") as fp: #Preallocate the local file to its full size
                        if big_size > 0:
                            fp.seek(big_size)
                            fp.write("\0")

            if ii%2==1 or not double_it:
                next_name = "big{0}.{1}.txt".format(big_size,(i+1)%n)
                logging.info("Transferring {0}".format(next_name))

                sleep_time=5.0
                for j in xrange(50):
                    if storage.file_exists(next_name):
                        break
                    logging.info("Waiting for '{0}' to exist. Will sleep {1}".format(next_name,sleep_time))
                    time.sleep(sleep_time)
                    sleep_time = min(60.0,sleep_time*1.1)
                assert storage.file_exists(next_name), "{0} still doesn't exist".format(next_name)
                
                t2 = time.time()
                with storage.open_read(next_name) as file_name:
                    pass
                mbps2 = _mbps(big_size, time.time()-t2)
                logging.info("transfers Mbps={0}".format(mbps2))

                return Mbps2
            return None

        mbps_list = map_reduce(xrange(nn),
                        mapper=mapper,
                        reducer=lambda sequence:[x for x in sequence if x is not None],
                        name="big_filename.{0}{1}".format(n,".x2" if double_it else ""),
                        runner=runner)

        return mbps_list

    def zzztest_big_files_slow_down(self): #!!! need to update
        logging.info("test_big_files_slow_down")
        from onemil.AzureP2P import AzureP2P, ip_address
        from fastlmm.association.tests.test_single_snp_all_plus_select import mf_to_runner_function
        def closure():
            unique_name = ip_address()
            root = r"\\{0}\scratch".format(unique_name)
            return unique_name, root
        storage = AzureP2P("/flstor/azurecopypy/test/bfsd",local_lambda=closure)


        big_size = int(4e9)
        update_python_path = "once" #!!! should be "once"
        pool_id_list=["chrom4"]
        AzureBatch(task_count=1,pool_id_list=pool_id_list,update_python_path=update_python_path)
        path_list = ["/flstor/azurecopypy/test/bfsd","/chrom1/azurecopypy/test/bfsd","/chrom4/azurecopypy/test/bfsd"]

        azure_path_list = self._upload_big_file(big_size,path_list)
        while True:
            for n in [3]:#1,2,4,8]: #range(10,24,2)+
                runner = AzureBatch(task_count=n,pool_id_list=pool_id_list,update_python_path="no")
                #runner = Local()
                mbps_list = self._big_files_slow_down_internal(self.container,big_size,n,azure_path_list,runner,storage)
                logging.info("{0}\t{1}".format(n,mbps_list))
         
    def _upload_big_file(self,big_size,path_list):
        def mapper(path):
            azure_path = "{0}/big{1}.txt".format(path,big_size)
            if not self.container.file_exists(azure_path):
                logging.info("Uploading '{0}'".format(azure_path))
                temp_parent = os.path.join(tempfile.gettempdir(), format(hash(os.times())))
                file_name = temp_parent+azure_path
                logging.info("Creating {0}".format(file_name))
                pstutil.create_directory_if_necessary(file_name)
                with open(file_name,"wb") as fp: #Preallocate the local file to its full size
                    if big_size > 0:
                        fp.seek(big_size)
                        fp.write("\0")
                with _file_transfer_reporter("'{0}'".format(azure_path),big_size) as updater:
                    self.container.upload(file_name,azure_path,updater=updater)
                os.remove(file_name)
            return azure_path
        azure_path_list = map_reduce(path_list,
                        mapper=mapper,
                        runner=LocalMultiThread(len(path_list))
                        )
        return azure_path_list

    @staticmethod
    def _big_files_slow_down_internal(container,big_size,n,azure_path_list,runner,storage):
        def mapper(i):
            azure_path = azure_path_list[i%len(azure_path_list)]
            short_name = "big{0}.{1}.txt".format(big_size,i)
            if storage.file_exists(short_name):
                storage.remove(short_name)
            with storage.open_write(short_name,size=big_size) as file_name:
                logging.info("Downloading {0}".format(azure_path))
                with _file_transfer_reporter("Downloading {0}".format(azure_path),big_size) as updater:
                    t0 = time.time()
                    container.download(azure_path,file_name,updater=updater)
                    mbps0 = _mbps(big_size, time.time()-t0)
            return mbps0

        mbps_list = map_reduce(xrange(n),
                        mapper=mapper,
                        name="big_files_slow_down",
                        runner=runner)

        return mbps_list

    def zzztest_storage_account_creation(self): #!!! need to update
        from azure.common.credentials import UserPassCredentials
        from azure.mgmt.resource import ResourceManagementClient
        from azure.mgmt.storage import StorageManagementClient
        from azure.storage import CloudStorageAccount
        from azure.storage.blob.models import ContentSettings
        import getpass

        username, subscription_id = [s.strip() for s in open(os.path.expanduser("~")+"/azurebatch/account.txt").xreadlines()]
        print "Azure password"
        password = getpass.getpass()
        credentials = UserPassCredentials(username, password)
        resource_client = ResourceManagementClient(credentials, subscription_ids)
        storage_client = StorageManagementClient(credentials, subscription_id)
        resource_client.resource_groups.create_or_update( 'my_resource_group', { 'location':'westus' } )
        async_create = storage_client.storage_accounts.create( 'my_resource_group', 'my_storage_account', { 'location':'westus', 'account_type':'Standard_LRS' } )

        async_create.wait()
        storage_keys = storage_client.storage_accounts.list_keys('my_resource_group', 'my_storage_account')
        storage_keys = {v.key_name: v.value for v in storage_keys.keys}
        storage_client = CloudStorageAccount('my_storage_account', storage_keys['key1'])
        blob_service = storage_client.create_block_blob_service()
        blob_service.create_container('my_container_name')
        blob_service.create_blob_from_bytes( 'my_container_name', 'my_blob_name', b'<center><h1>Hello World!</h1></center>',
            content_settings=ContentSettings('text/html') )
        print(blob_service.make_blob_url('my_container_name', 'my_blob_name'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #from onemil.azure_copy import TestAzureShardContainer #!!! having this here lets us run on Azure, but stops us from using breakpoints

    suites = unittest.TestLoader().loadTestsFromTestCase(TestAzureShardContainer)


    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=True) #!!!by default should be false
        r.run(suites)
    else: #runner test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest
        runner = Local() #LocalMultiProc(taskcount=22,mkl_num_threads=5,just_one_process=True)
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done")
