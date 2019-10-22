import numpy as np
import random
import os
import shutil
import tempfile
import logging
import unittest
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalMultiThread
from onemil.file_cache import AzureStorage,PeerToPeer,FileCache,LocalCache
import time
from onemil.file_cache import DibLib
from contextlib import contextmanager
import threading

try:
    import azure.batch.models as batchmodels
    import azure.storage.blob as azureblob
    azure_ok = True
except Exception, exception:
    logging.warning("Can't import azure, so won't be able to clusterize to azure")
    azure_ok = False

if azure_ok:
    import onemil.azurehelper as commonhelpers #!!! is this the best way to include the code from the Azure python sample's common.helper.py?
    import azure.batch.batch_service_client as batch 
    import azure.batch.batch_auth as batchauth 
    from onemil.blobxfer import run_command_string as blobxfer #https://pypi.io/project/blobxfer/
    from onemil.blobxfer import compute_md5_for_file_asbase64

def path_join(*p):
    result = os.path.normpath("/".join(p)).replace('\\','/')
    return result

def ip_address_local():
    '''
    A function the returns 1. the ip address of the machine on which it is run and 2. a Windows file share on that machine called 'scratch'

    It is used by :class:`AzureP2P` to retrieve a local place from which peers may copy files.
    '''
    unique_name = ip_address()
    root = r"\\{0}\scratch".format(unique_name)
    return unique_name, root

def ip_address_pid_local(): #!!! compare with ip_address_pid
    '''
    A function the returns 1. the ip address of the machine on which it is run and 2. a Windows file share on that machine called 'scratch\\\<pid>',
    where <pid> is the current process id.

    It is used by :class:`AzureP2P` to retrieve a local place from which peers may copy files. The addition of the <pid> allows multiple processes
    on the same machine to have separate storage from each other. This make testing :class:`AzureP2P` easier.
    '''
    ip = ip_address()
    pid = os.getpid()
    tid = threading.current_thread().ident
    root = r"\\{0}\scratch\{1}.{2}".format(ip,pid,tid)
    return "{0}.{1}.{2}".format(ip,pid,tid), root


class AzureP2P(FileCache):
    '''
    A class that subclasses :class:`FileCache` to provide peer-to-peer file sharing backed up by Azure Storage.
    an Azure batch account.

    **Constructor:**
        :Parameters: * **azure_root** (*string*) -- The path on Azure storage under which data will be stored. The form of the path is:
                           /STORAGEACCOUNT/CONTAINER/morepath.
                     * **local_lambda** (*a zero-augment lambda*) -- When called, tells were to store data locally. See :func:`ip_address_local`.
                     * **leave_space** (*integer*) -- (default 0) Tells the minimum amount of local storage space that should be kept free.
                     * **storage_credential** (*:class:`StorageCredential`*) -- (default '~/azurebatch/cred.txt') Keys and names of Azure Storage and AzureBatch
                           accounts to use.
                     * **subpath** (*string*) -- (default '.') Additional path to append to **azure_root**.

    **All the methods of FileCache plus these:**
    '''

    def __init__(self, azure_root, local_lambda, leave_space=0,storage_credential=None, subpath="."):
        super(AzureP2P, self).__init__()
        self.azure_root = azure_root
        self.subpath = subpath
        self.local_lambda = local_lambda  #When called, returns a unique name (e.g. 172.10.12.3) and root storage (e.g. \\172.10.12.3\scratch). Expect repeated called to return the same two values #!!! document this
        self.leave_space = leave_space
        self.storage_credential = storage_credential

        self.azure_storage = AzureStorage(path_join(self.azure_root,"storage",subpath),local_lambda=self.local_lambda,storage_credential=self.storage_credential)

        self.dib_directory = AzureStorage(path_join(self.azure_root,"dibs",subpath),local_lambda=self.local_lambda,storage_credential=storage_credential)

        def storage_lambda():
            unique_name, root = self.local_lambda()
            return unique_name, root + path_join(self.azure_root,"storage",subpath)
        azure_directory = AzureStorage(path_join(self.azure_root,"directory",subpath),local_lambda=self.local_lambda,storage_credential=self.storage_credential)
        self.file_share = PeerToPeer(directory=azure_directory,local_lambda=storage_lambda, leave_space=self.leave_space)

    def _simple_join(self,path):
        assert not self.azure_storage.file_exists(path), "Can't treat an existing file as a directory"
        return AzureP2P(self.azure_root, local_lambda=self.local_lambda, leave_space=self.leave_space,storage_credential=self.storage_credential,subpath=path_join(self.subpath,path))

    def __repr__(self):
            return "{0}('{1}')".format(self.__class__.__name__,self.name)

    @property
    def name(self):
        return self.azure_root

    def _simple_file_exists(self,simple_file_name):
        return self.azure_storage._simple_file_exists(simple_file_name)

    @contextmanager
    def _simple_open_write(self,simple_file_name,size=0,updater=None):
        subhandle_as = self.azure_storage.open_write(simple_file_name,size=size,updater=updater)
        subhandle_as_file_name = subhandle_as.__enter__()

        if self.file_share._simple_file_exists(simple_file_name):
            logging.warn("The AzureStorage doesn't already have the file that is being written, but the PeerToPeer does, so removing it from the PeerToPeer. {0},'{1}'".format(self.file_share,simple_file_name))
            self.file_share._simple_remove(simple_file_name)
        subhandle_fs = self.file_share.open_write(simple_file_name,size=size,updater=updater)
        subhandle_fs_file_name = subhandle_fs.__enter__()

        assert os.path.normpath(subhandle_fs_file_name) == os.path.normpath(subhandle_as_file_name), "Expect that the two ways of distributing files to agree on the local file name"

        yield subhandle_fs_file_name

        subhandle_fs.__exit__(None,None,None)
        subhandle_as.__exit__(None,None,None) #This one needs to be last because it sets the file date

    @contextmanager
    def _simple_open_read(self,simple_file_name,updater=None):

        # We assume that we could use either sub-storage and that the only problem
        # that could happen is that the file_share's remote machine containing the "main"
        # file share copy could have been recycled and thus the file would be missing.

        is_ok = False
        try:
            subhandle1 = self.file_share._simple_open_read(simple_file_name) #!!!should file share be self-repairing. If the "main" is gone, pick one of the others
            subhandle1_file_name = subhandle1.__enter__()
            is_ok = True
        except Exception, e:
            logging.info("AzureP2P - machine-to-machine copy of '{0}' failed, so reading from AzureStorage. Exception='{1}'".format(simple_file_name,e))
        if is_ok:
            yield subhandle1_file_name
            subhandle1.__exit__(None,None,None)
            return

        #We are now in a situation in which multiple readers have failed to find the file in the PeerToPeer. We would like one of them to
        #download from AzureStorage, while the others wait. When that one has finished, the others can then get it from PeerToPeer.

        #Dib the file and see if you are 1st.
        # If so, double check that PeerToPeer isn't there now ((in case someone fixed everything already) if it is, use it) otherwise download and share.
        # If not wait until the first dib is gone, then clear your dib and use the file share.

        unique_name = self.local_lambda()[0]
        dib_path = self.dib_directory.join(simple_file_name)
        dir_path = self.file_share.directory.join(simple_file_name)
        dib_lib = DibLib(unique_name,dib_path,dir_path,"azure_storage_dib")
        status = dib_lib.wait_for_turn()
        logging.info("status is '{0}'".format(status))
        if status == 'fixed':
            logging.info("After waiting for someone else to fix the problem, can now read the file with PeerToPeer")
            read_handle = self.file_share._simple_open_read(simple_file_name) #!!!should file share be self-repairing. If the "main" is gone, pick one of the others
            yield read_handle.__enter__()
            read_handle.__exit__(None,None,None)
            dib_lib.remove_dibs()
            return
        elif status == 'azure':
            is_ok = False
            try:
                logging.info("Before I try azure, let's try file_share again")
                read_handle = self.file_share._simple_open_read(simple_file_name) #!!! should file share be self-repairing. If the "main" is gone, pick one of the others
                file_name2 = read_handle.__enter__()
                is_ok = True
            except Exception, e:
                logging.info("2nd try of reading from PeerToPeer failed with message '{0}'".format(e.message))
            if is_ok:
                yield file_name2
                read_handle.__exit__(None,None,None)
                dib_lib.remove_dibs()
                return

                
            logging.info("downloading from Azure")
            read_handle = self.azure_storage._simple_open_read(simple_file_name)
            file_name3 = read_handle.__enter__()
            self._simple_register_with_peer_to_peer(simple_file_name,file_name3)
            yield file_name3
            read_handle.__exit__(None,None,None)
            dib_lib.remove_dibs()
            return
        else:
            raise Exception("Don't know status '{0}'".format(status))
            
        
    def _simple_register_with_peer_to_peer(self, simple_file_name, local_file_name):
        #Removes any current peer-to-peer entry and adds a new one
        temp_name = local_file_name+"."+format(hash(os.times()))+".temp"
        os.rename(local_file_name,temp_name) #Rename so that this remove doesn't remove it
        if self.file_share._simple_file_exists(simple_file_name):
            self.file_share._simple_remove(simple_file_name)
        with self.file_share._simple_open_write(simple_file_name,size=0) as file_name: #size 0 because we already have space allocated.
            os.rename(temp_name,local_file_name) #Put file back
            assert os.path.normpath(local_file_name) == os.path.normpath(file_name), "Expect that the two ways of distributing files to agree on the local file name"

    def _simple_getmtime(self,simple_file_name):
        return self.azure_storage._simple_getmtime(simple_file_name)

    def _simple_rmtree(self, log_writer=None):

        #If they all share a common directory, kill that directory
        #why this could be a bad idea: What if the azure_shard_container's are different or if there is another directory there of interest
        #why this could be a great idea: If this is top-level Azure Storage container, deleting will take a second instead of an hour.
        if isinstance(self.file_share.directory,AzureStorage):
            def drop_end(folder):
                return '/'.join(folder.split('/')[:-1])
            folder1=drop_end(self.file_share.directory.folder)
            folder2=drop_end(self.dib_directory.folder)
            folder3=drop_end(self.azure_storage.folder)
            if folder1==folder2 and folder2==folder3:
                if log_writer is not None: log_writer("Fast rmtreeing '{0}'".format(folder1))
                self.azure_storage.azure_shard_container.rmtree(folder1)
                return

        self.file_share._simple_rmtree(log_writer=log_writer)
        self.azure_storage._simple_rmtree(log_writer=log_writer)
        self.dib_directory._simple_rmtree(log_writer=log_writer)

    def _simple_remove(self,simple_file_name,log_writer=None):
        if self.file_share._simple_file_exists(simple_file_name):
            self.file_share._simple_remove(simple_file_name)
        self.azure_storage._simple_remove(simple_file_name,log_writer=log_writer)

    def _simple_walk(self):
        return self.azure_storage._simple_walk()


    def azure_storage_only(self,path=None,log_writer=None):
        '''
        Remove everything except the AzureStorage copy of the file system
        '''
        self.dib_directory.rmtree() #Remove everything from the dib directory
        self.file_share._remove_local_if_any(path)
        self.file_share.directory.rmtree(path,log_writer=log_writer)

    def remove_from_azure_storage(self,path=None,log_writer=None):
        '''
        Remove everything from AzureStorage (local copies will remain, but will be ignored)
        '''
        if path is None:
            self.rmtree(log_writer=log_writer) #This will remove azure_storage and directory (and the dibs, again)
        elif self.azure_storage.file_exists(path):
            self.azure_storage.remove(path)
        self.azure_storage_only(path,log_writer=log_writer) #This will remove any local files (and the dibs, again)

#!!! move this testing closer to file_cache.py
class TestAzureP2P(unittest.TestCase):

    def test_local_file(self):
        logging.info("test_local_file")

        temp_dir = self._temp_dir()
        def storage_closure():
            return LocalCache(temp_dir)
        self._write_and_read(storage_closure())
        self._distribute(storage_closure)
    
    def test_file_share_traditional(self):
        logging.info("test_file_share_traditional")

        #Everyone shares a directory but has their own storage
        directory = self._temp_dir()+"/directory"
        def storage_closure():
            temp_dir = self._temp_dir() #increments self.count
            count = self.count
            def local_lambda():
                unique_name, root = "{0}.{1}".format(os.environ['COMPUTERNAME'],count), temp_dir+"/storage"
                return unique_name, root
            storage = PeerToPeer(directory=directory,local_lambda=local_lambda)
            return storage

        self._write_and_read(storage_closure())
        self._distribute(storage_closure)

    def test_azure_storage(self):
        logging.info("test_azure_storage")

        def storage_closure():
            default_shared_dir_lambda = self._temp_dir()
            storage = AzureStorage("/flstor/testazurep2p/azure_storage",local_lambda=lambda:(None,default_shared_dir_lambda))
            return storage

        self._write_and_read(storage_closure())
        self._distribute(storage_closure)
        
    def test_file_share_via_azure(self):
        logging.info("test_file_share_via_azure")

        #Everyone shares a directory but has their own storage
        directory = self._temp_dir()+"/azure"
        def storage_closure():
            temp_dir = self._temp_dir() #increments self.count
            count = self.count
            def local_lambda():
                unique_name, root = "{0}.{1}".format(os.environ['COMPUTERNAME'],count), temp_dir+"/storage"
                return unique_name, root
            storage = PeerToPeer(directory=AzureStorage("/flstor/testazurep2p/fileshare/directory",local_lambda=lambda:(None,directory)),local_lambda=local_lambda)
            return storage

        self._write_and_read(storage_closure())
        self._distribute(storage_closure)

    def test_azure_p2p(self):
        logging.info("test_azure_p2p")

        def storage_closure():
            self.count += 1
            count = self.count
            def local_lambda():
                unique_name = "{0}.{1}".format(os.environ['COMPUTERNAME'],count)
                root = r"\\{0}\scratch\{1}".format(os.environ['COMPUTERNAME'],count)
                return unique_name, root
            storage = AzureP2P("/flstor/testazurep2p/p2p",local_lambda=local_lambda)
            return storage


        self._write_and_read(storage_closure())
        self._distribute(storage_closure)

    def test_azure_p2p_robust(self):
        
        logging.info("test_azure_p2p_robust")

        def storage_closure():
            self.count += 1
            count = self.count
            def local_lambda():
                unique_name = "{0}.{1}".format(os.environ['COMPUTERNAME'],count)
                root = r"\\{0}\scratch\{1}".format(os.environ['COMPUTERNAME'],count)
                return unique_name, root
            storage = AzureP2P("/flstor/testazurep2p/p2p_robust",local_lambda=local_lambda)
            return storage


        #If we knock out the main copy, it reverts to AzureStorage
        storage1 = storage_closure()
        storage2 = storage_closure()
        storage1.rmtree()
        storage1.save("a/b/c.txt","Hello")
        shutil.rmtree(storage1.file_share.local_lambda()[1]) #Removing the main file share copy
        assert storage2.load("a/b/c.txt")=="Hello" #Should load from Azure, now
        assert storage2.load("a/b/c.txt")=="Hello" #Should load from local cache
        storage3 = storage_closure()
        assert storage3.load("a/b/c.txt")=="Hello" #Should load from file share again
        
        #If we knock out the main copy, but there are other copies, it promotes one of them to 'main'
        storage1.rmtree()
        storage1.save("a/b/c.txt","Hello")
        assert storage2.load("a/b/c.txt")=="Hello" #Should load from Azure, now
        shutil.rmtree(storage1.file_share.local_lambda()[1]) #Removing the main file share copy
        assert storage3.load("a/b/c.txt")=="Hello"

    def test_azure_p2p_multiproc(self):
        from pysnptools.util.mapreduce1.mapreduce import map_reduce
        import threading

        logging.info("test_azure_p2p_multiproc")
        runner = LocalMultiProc(3,just_one_process=False)# Local()
        #runner = LocalMultiThread(3,just_one_process=False)

        storage = AzureP2P("/flstor/testazurep2p/multiproc",local_lambda=ip_address_pid_local)
        #storage = AzureStorage("test/multiproc",default_shared_dir_lambda=closure)
        #storage = PeerToPeer(directory=AzureStorage("test/multiproc/directory",default_shared_dir_lambda=lambda:closure()+"/azure"),
        #                    storage_lambda=lambda:closure()+"/storage",unique_name=lambda:"{0}.{1}".format(os.environ['COMPUTERNAME'],os.getpid()))
        storage.rmtree()
        storage.save("a/b/c.txt","Hello")
        shutil.rmtree(storage.file_share.local_lambda()[1]) #Removing the main file share copy

        def mapper_closure(id):
            assert storage.load("a/b/c.txt")=="Hello"
            return True

        result = map_reduce(xrange(4),
                    mapper=mapper_closure,
                    runner=runner
                    )

        logging.info(result)
        logging.info("done with test")

    @staticmethod
    def file_name(self,testcase_name):
            temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
            if os.path.exists(temp_fn):
                os.remove(temp_fn)
            return temp_fn
    tempout_dir = "tempout/one_milp2p"

    @classmethod
    def setUpClass(self):
        self.temp_parent = os.path.join(tempfile.gettempdir(), format(hash(os.times())))
        self.count = 0

    def _temp_dir(self):
        self.count += 1
        return "{0}/{1}".format(self.temp_parent,self.count)

    @classmethod
    def tearDownClass(self):
        not os.path.exists(self.temp_parent) or shutil.rmtree(self.temp_parent)

    def _is_error(self,lambda0):
        try:
            lambda0()
        except Exception, e:
            logging.debug(e.message)
            return True
        return False

    def _len(self,sequence):
        len = 0
        for item in sequence:
            len += 1
        return len
        
    def _write_and_read(self,storage):
        test_storage = storage.join('test_snps') #!!!How to induce an error: create a 'test_snps' file at the top level then try to create an empty directory with the same name


        #Clear the directory
        test_storage.rmtree()
        #Rule: After you clear a directory, nothing is in it
        assert 0 == self._len(test_storage.walk())
        assert not test_storage.file_exists("test.txt")
        assert not test_storage.file_exists("main.txt/test.txt")
        assert not test_storage.file_exists(r"main.txt\test.txt")
        assert self._is_error(lambda : test_storage.file_exists("test.txt/")) #Can't query something that can't be a file name
        assert self._is_error(lambda : test_storage.file_exists("../test.txt")) #Can't leave the current directory
        assert self._is_error(lambda : test_storage.file_exists(r"c:\test.txt")) #Can't leave the current directory

        #Rule: '/' and '\' are both OK, but you can't use ':' or '..' to leave the current root.
        assert 0 == self._len(test_storage.walk())
        assert self._is_error(lambda : 0 == self._len(test_storage.walk("..")))
        assert 0 == self._len(test_storage.walk("..x"))
        assert 0 == self._len(test_storage.walk("test.txt")) #This is ok, because test.txt doesn't exist and therefore isn't a file
        assert 0 == self._len(test_storage.walk("a/b"))
        assert 0 == self._len(test_storage.walk("a\\b")) #Backslash or forward is fine
        assert self._is_error(lambda : len(test_storage.walk("/"))) #Can't start with '/'
        assert self._is_error(lambda : len(test_storage.walk(r"\\"))) #Can't start with '\'
        assert self._is_error(lambda : len(test_storage.walk(r"\\computer1\share\3"))) #Can't start with UNC

        #Clear the directory, again
        test_storage.rmtree()
        assert 0 == self._len(test_storage.walk())
        test_storage.rmtree("main.txt")
        assert 0 == self._len(test_storage.walk("main.txt"))
        assert 0 == self._len(test_storage.walk())


        #Write to it.
        assert self._is_error(lambda : test_storage.save("../test.txt"))
        test_storage.save("main.txt/test.txt","test\n")
        #Rule: It's an error to write to a file or directory that already exists
        assert self._is_error(lambda : test_storage.save("main.txt")) 
        assert self._is_error(lambda : test_storage.save("main.txt/test.txt")) 
        assert self._is_error(lambda : list(test_storage.walk("main.txt/test.txt"))), "Rule: It's an error to walk a file (but recall that it's OK to walk a folder that doesn't exist)"

        #It should be there and be a file
        assert test_storage.file_exists("main.txt/test.txt")
        file_list = list(test_storage.walk())
        assert len(file_list)==1 and file_list[0] == "main.txt/test.txt"
        file_list2 = list(test_storage.walk("main.txt"))
        assert len(file_list2)==1 and file_list2[0] == "main.txt/test.txt"
        assert self._is_error(lambda : test_storage.join("main.txt/test.txt")) #Can't create a directory where a file exists
        assert self._is_error(lambda : list(test_storage.walk("main.txt/test.txt"))) #Can't create a directory where a file exists
        assert self._is_error(lambda : test_storage.rmtree("main.txt/test.txt")) #Can't delete a directory where a file exists

        #Read it
        assert test_storage.load("main.txt/test.txt")=="test\n"
        assert test_storage.file_exists("main.txt/test.txt")
        assert self._is_error(lambda : test_storage.load("main.txt"))  #This is an error because main.txt is actually a directory and they can't be opened for reading

        #Remove it
        test_storage.remove("main.txt/test.txt")
        assert self._is_error(lambda : test_storage.remove("main.txt/test.txt")) #Can't delete a file that doesn't exist
        assert not test_storage.file_exists("main.txt/test.txt")
        assert 0 == self._len(test_storage.walk())
        assert 0 == self._len(test_storage.walk("main.txt"))
        assert 0 == self._len(test_storage.walk("main.txt/test.txt")) #Now allowed.

        #  writing zero length files is OK
        #  File share has a special file called "main.txt". Can we mess things up by using 'main.txt' as a directory name, too.
        #It's OK to write to a file in a directory that used to exist, but now has no files.
        test_storage.save("main.txt","")
        assert test_storage.file_exists("main.txt")
        file_list = list(test_storage.walk())
        assert len(file_list)==1 and file_list[0] == "main.txt"
        assert test_storage.load("main.txt") == ""
        assert test_storage.file_exists("main.txt")

        #Can query modified time of file. It will be later, later.
        assert self._is_error(lambda : test_storage.getmtime("a/b/c.txt")), "Can't get mod time from file that doesn't exist"
        test_storage.save("a/b/c.txt","")
        m1 = test_storage.getmtime("a/b/c.txt")
        assert self._is_error(lambda : test_storage.getmtime("a/b")), "Can't get mod time from directory"
        assert test_storage.getmtime("a/b/c.txt") == m1, "expect mod time to stay the same"
        assert test_storage.load("a/b/c.txt") == ""
        assert test_storage.getmtime("a/b/c.txt") == m1, "reading a file doesn't change its mod time"
        test_storage.remove("a/b/c.txt")
        assert self._is_error(lambda : test_storage.getmtime("a/b/c.txt")), "Can't get mod time from file that doesn't exist"
        time.sleep(1) #Sleep one second
        test_storage.save("a/b/c.txt","")
        test_storage.save("a/b/d.txt","")
        assert test_storage.getmtime("a/b/c.txt") > m1, "A file created later (after a pause) will have a later mod time"
        assert test_storage.getmtime("a/b/d.txt") > m1, "A file created later (after a pause) will have a later mod time"
        assert test_storage.getmtime("a/b/d.txt") >= test_storage.getmtime("a/b/c.txt"), "A file created later (with no pause) will have a later or equal mod time"

        logging.info("done")

    # Look for code that isn't covered and make test cases for it
    #Distributed testing
    # What happens if you don't close an open_write? Undefined
    
    def _distribute(self,storage_lambda):
        storage1 = storage_lambda()
        storage2 = storage_lambda()
        #clear everything on #1
        storage1.rmtree()
        #1 and #2 agree nothing is there
        assert 0 == self._len(storage1.walk())
        assert 0 == self._len(storage2.walk())
        #write on #1
        storage1.save("a/b/c.txt","Hello")
        #read on #2
        assert storage2.load("a/b/c.txt")=="Hello"
        #read on #2, again and manually see that it caches
        assert storage2.load("a/b/c.txt")=="Hello"
        #remove on #1
        storage1.remove("a/b/c.txt")
        #assert not exists on #2
        assert not storage2.file_exists("a/b/c.txt")
        #write on #1
        storage1.save("a/b/c.txt","Hello")
        #read on #2
        assert storage2.load("a/b/c.txt")=="Hello"
        #remove on #1
        storage1.remove("a/b/c.txt")
        #write something different on #1
        storage1.save("a/b/c.txt","There")
        #read on #2 and see that it is different.
        assert storage2.load("a/b/c.txt")=="There"

    

def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAzureP2P))
    #test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))
    return test_suite

if __name__ == '__main__':
    from msgen_cli.malibucommon import pause_and_display
    import msgen_cli.datatransfer as datatransfer
    from onemil.AzureP2P import getTestSuite, TestAzureP2P

    logging.basicConfig(level=logging.INFO)
    suites = getTestSuite()

    if True: #Standard test run
        r = unittest.TextTestRunner(failfast=False)#!!! should be false
        r.run(suites)
    else: #runner test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest
        #runner = LocalMultiProc(taskcount=22,mkl_num_threads=5,just_one_process=True)
        runner = Local()
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done")
