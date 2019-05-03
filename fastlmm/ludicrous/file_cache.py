import os
import pysnptools.util as pstutil
import psutil
import glob
import random
import shutil
import logging
import glob
import time
import datetime
import random
import pytz
import io
import multiprocessing
import re
import shutil
import subprocess
from contextlib import contextmanager
import sys

def _format_delta(delta):
    return datetime.timedelta(seconds=delta)

def Mbps(size, delta):
    return size * 8 / delta / 1e6

def MbpsStr(t0, size, total=0):
    delta = time.time()-t0
    Mbps0 = Mbps(size, delta) if delta > 0 else 0
    percent = float(size)/total if total > 0 else 1
    left = delta/percent*(1-percent) if total > 0 else 0
    return "{0:0.2f}, {1:0.1f}%, left={2}".format(Mbps0,percent*100,_format_delta(left))


#!!! move this to util file
@contextmanager
def log_in_place(name, level, time_lambda=time.time, show_log_diffs=False):
    '''
        Create an one-argument lambda to write messages to. They will appear on the same line.

        Example::

            with log_in_place("creating job_id_etc", logging.INFO) as log_writer:            
                job_id_etc_list = []
                for index,(distributable,pool_id,name_in) in enumerate(izip(distributable_list,self.pool_id_list,name_list)):
                   log_writer(name_in)
                   job_id_etc = self._setup_job([distributable],pool_id,name_in)[0]
                   job_id_etc_list.append(job_id_etc)

    '''
    #!!! what if logging messages aren't suppose to go to stdout?
    t_wait = time_lambda()
    last_len = [0] #We have to make this an array so that the value is by reference.
    last_message_hash = [None]
    line_end = '\r'

    def writer(message):
        if logging.getLogger().level > level:
            return
        time_str = str(datetime.timedelta(seconds=time_lambda()-t_wait))
        if '.' in time_str:
            time_str = time_str[:time_str.index('.')+3] #Time to the 1/100th of a sec
        s = "{0} -- time={1}, {2}".format(name,time_str,message)
        if show_log_diffs:
            message_hash = hash(message)
            if message_hash !=  last_message_hash[0] and last_message_hash[0] is not None:
                sys.stdout.write('\n')
            last_message_hash[0] = message_hash
        sys.stdout.write("{0}{1}\r".format(s," "*max(0,last_len[0]-len(s)))) #Pad with spaces to cover up previous message
        last_len[0] = len(s)
        

    yield writer

    if logging.getLogger().level > level:
        return
    sys.stdout.write("\n")                

@contextmanager
def progress_reporter(name,size=None,updater=None):
    '''
    If an update is given, we use that. Otherwise, we create our own.
    '''
    if updater is None:
        bytes_so_far = [0] #We have to make this an array so that the value is by reference.
        t0 = time.time()

        with log_in_place(name, logging.INFO) as writer:
            def updater2(byte_size):
                bytes_so_far[0] += byte_size
                writer("Mbps={0}".format(MbpsStr(t0,bytes_so_far[0],size)))
            yield updater2
    else:
        yield updater


@contextmanager
def multiopen(open_lambda, input_list):
    handle_list = [open_lambda(input) for input in input_list]        # Open the related inputs
    list_to_yield = [handle.__enter__() for handle in handle_list]    # Get the list to yield
    yield list_to_yield                                               # yield it
    for handle in handle_list:                                        # Close them
        handle.__exit__(None,None,None)

def path_join(*p):
    result = os.path.normpath("/".join(p)).replace('\\','/')
    return result


class FileCache(object):
    '''
    An abstract class for providing storage.
    '''

    def __init__(self):
        super(FileCache, self).__init__()

    def join(self,path):
        '''
        The :class:`FileCache` created by appending a path to the current :class:`FileCache`.

        '''
        head = self._normpath(path)
        if head is None:
            return self
        return self._simple_join(head)

    def walk(self,path=None):
        '''
        Generates the paths of the files in the path, relative to the path. It is OK if there are no files.
        '''
        for item in self.join(path)._simple_walk():
            yield item if path is None else path + "/" + item
            

    def rmtree(self,path=None,log_writer=None):
        '''
        Delete all files this under this path. It can not be a file. It is OK if there are no files.
        '''
        return self.join(path)._simple_rmtree(log_writer=log_writer)

    def file_exists(self,file_name):
        '''
        Tells if there a file with this name. (A directory with the name doesn't count.)
        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_file_exists(simple_file)

    def open_read(self,file_name,updater=None):
        '''
        Used with a 'with' statement to produce a local copy of the file.

        Example::

            with self.open_read(file_name,updater=updater) as local_file_name:
                with open(local_file_name,"r") as fp:
                    line = fp.readline()
            return line

        '''
        directory, simple_file_name = self._split(file_name)
        return directory._simple_open_read(simple_file_name,updater=updater)

    def open_write(self,file_name,size=0,updater=None):
        '''
        Used with a 'with' statement to produce a local file name that will be 'uploaded'
        when the 'with' statement is exited.
    
        if 'size' is given, an error will be thrown immediately if local storage doesn't
        have room for that many bytes.

        Example::

            with self.open_write(file_name,size=size,updater=updater) as local_file_name:
                with open(local_file_name,"w") as fp:
                    fp.write(contents)


        '''
        directory, simple_file_name = self._split(file_name)
        return directory._simple_open_write(simple_file_name,size=size,updater=updater)


    def remove(self,file_name,log_writer=None):
        '''
        Remove a file from storage. It is an error to remove a directory this way.
        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_remove(simple_file,log_writer=log_writer)

    def save(self, file_name, contents,size=0,updater=None):
        '''
        Write a string to a file in storage.
        '''
        with self.open_write(file_name,size=size,updater=updater) as local_file_name:
            with open(local_file_name,"w") as fp:
                fp.write(contents)

    def load(self, file_name,updater=None):
        '''
        Returns the contents of a file in storage as a string.
        '''
        with self.open_read(file_name,updater=updater) as local_file_name:
            with open(local_file_name,"r") as fp:
                line = fp.readline()
        return line

    def getmtime(self,file_name):
        '''
        Return the modified date of the file in storage.
        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_getmtime(simple_file)



    @property
    def name(self):
        '''
        A nice name of this storage location.
        '''
        return "FileCache"

    def _split(self,file_name):
        head, tail = os.path.split(file_name)
        if not tail:
            assert tail, "Expect a file name"
        if head == "":
            return self, tail
        else:
            return self.join(head), tail
        
    def _normpath(self,path):
        if path is None:
            return None
        head = os.path.normpath(path).replace('\\','/')
        if head == ".":
            return None
        assert os.path.splitunc(head)[0] == "", "Should not be UNC"
        assert os.path.splitdrive(head)[0] == "", "Should have a drive"
        assert head != ".." and not head.startswith("../"), "Should not leave parent"
        assert not head.startswith("/"), "Should not start with '/'"
        return head

    def _at_least_one(self,sequence):
        for item in sequence:
            return True
        return False

    @staticmethod
    def _create_directory(local):
        if os.path.exists(local):
            if os.path.isfile(local):
                os.remove(local)
            else:
                shutil.rmtree(local)
        directory_name = os.path.dirname(local)
        if os.path.exists(directory_name) and os.path.isfile(directory_name):
            os.remove(directory_name)
        pstutil.create_directory_if_necessary(local,isfile=True)

class FileShare(FileCache): #!!!rename FileShare to PeerToPeer (and put all these classes in their own file)
    #Later features:
    #   Limit the amount of copying for a given file
    #   Work even when multiple tasks on same node ask for the same file
    #   Get open_read,close to work with python With statement
    #   Have a way to clean up lost files
    def __init__(self,directory,local_lambda,leave_space=0):
        super(FileShare, self).__init__()
        if isinstance(directory,str):
            directory = LocalCache(directory)
        self.directory = directory
        self.local_lambda = local_lambda
        self.leave_space = leave_space
        assert not self.directory._simple_file_exists("main.txt"), "A directory cannot exist where a file already exists."

    def __repr__(self): 
        return "{0}('{1}','{2}')".format(self.__class__.__name__,self.directory,self.leave_space)

    @property
    def name(self):
        return self.directory

    def _simple_file_exists(self,simple_file_name):
        return self.directory.file_exists(simple_file_name+"/main.txt")

    def _far_file_sequence(self,dir_path):
        storage_list = [f for f in dir_path.walk() if self.copy_main_pattern.match(f)]
        random.shuffle(storage_list)
        
        for storage_item in storage_list:
            storage_path = dir_path.load(storage_item)
            yield storage_path, len(storage_list)

    copy_main_pattern = re.compile("^((main)|(copy_.*))\.txt$")
    copy_pattern = re.compile("^copy_.*\.txt$")

    def copy_time_stamp(self,storage_path,local_path):
        timestamp = os.path.getmtime(storage_path)
        with open(local_path, 'a'): #Touch the local file so that its mod time will be later than the remote time (see http://stackoverflow.com/questions/1158076/implement-touch-using-python)
            os.utime(local_path, (timestamp,timestamp))

    @contextmanager
    def _simple_open_read(self,simple_file_name,updater=None):

        logging.debug("open_read('{0}')".format(simple_file_name))
        #
        #Returns name of local file and locks that local until released
        #

        #Try to Get locally, return it if worked
        #Is there enough room? If not delete some files. If still not, fail
        #Ask origin for randomized list of copies (the last one will be the origin, but not all have to be included)
        #Try getting each copy. If all fail, then fail
        #Register local copy with the origin
        dir_path=self.directory.join(simple_file_name)
        unique_name, root = self.local_lambda()
        local_path = root + "/" + simple_file_name
        pstutil.create_directory_if_necessary(local_path,isfile=True)
        copy_name = "copy_{0}.txt".format(unique_name)
        main_path = self._robust_load_main(dir_path)
        file_size = os.path.getsize(main_path)

        if os.path.exists(local_path):
            if main_path==local_path or dir_path.file_exists(copy_name):
                logging.info("\tfound local")
                assert file_size == os.path.getsize(local_path), "Local file doesn't have the same size as the main file"
                yield local_path
                return
            else:
                logging.info("\tlost local found. Will remove")
                os.remove(local_path)
                pstutil.create_directory_if_necessary(local_path,isfile=True)

        dib_lib = DibLib(unique_name,dir_path,dir_path,"dibs")
        try:
            dib_lib.wait_for_turn()

            #Now we can copy. Choose a source at random
            for storage_path, storage_count in self._far_file_sequence(dir_path): # In a loop because in the future may want to handle copies being deleted.
                try: #If something goes wrong, try the next one
                    self._net_use(storage_path)
                    assert file_size == os.path.getsize(storage_path), "File to copy from doesn't have the same size as the main file"
                    if psutil.disk_usage(os.path.dirname(local_path)).free - file_size < self.leave_space: #clean up the directories
                        #Check every local file and if it is not in the directory (e.g. a temp file from other programs), remove it
                        local_dir = os.path.split(local_path)[0]
                        for other_file in os.listdir(local_dir):
                            full_file = local_dir + "/" + other_file
                            if os.path.isfile(full_file) and not self._simple_file_exists(other_file):
                                logging.info("\tNeed space and file isn't in directory so removing it. '{0}'".format(full_file))
                                os.remove(full_file)
                    logging.info("\tshutil.copyfile('{0}','{1}')".format(storage_path,local_path))
                    then = datetime.datetime.now()
                    shutil.copyfile(storage_path,local_path)
                    shutil.copystat(storage_path,local_path)
                    self.copy_time_stamp(storage_path,local_path)
                    dir_path.save(copy_name,local_path)
                    delta_sec = max((datetime.datetime.now()-then).total_seconds(),1.)
                    try: #The 'try' stops this logging message from getting a div by zero error some times
                        logging.info("Copy time is {0}. Copy speed is {1} Mbps".format(_format_delta(delta_sec), 8.0*float(file_size)/1e6/delta_sec))
                    except:
                        logging.info("Copy time is {0}. Copy speed can't be calculated Mbps".format(_format_delta(delta_sec)))
                    break
                except Exception, e:
                    if os.path.exists(local_path):
                        logging.warning("If a local file was created, but something went wrong (e.g. the source disappeared part way through the copying), so we remove it")
                        os.remove(local_path)
                    logging.warning("Ignore exception {0}".format(e))
            assert file_size == os.path.getsize(local_path), "File did not copy or did not copy completely."
            
        finally:
            dib_lib.remove_dibs()

        yield local_path

        assert dir_path._simple_file_exists("main.txt"), "File doesn't exist: '{0}'".format(path)

    @staticmethod
    def _net_use(path):
        if not path.startswith(r"\\"):
            return
        path = path.replace('/','\\')
        path = '\\'.join(path.split('\\')[:4])
        if True:
            subprocess.call(r'net use {0} "" /u:""'.format(path),shell=True)
        else:
            process = subprocess.Popen(r'net use {0} "" /u:""'.format(path), shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate() # wait for the process to terminate
            if ("Multiple connections to a server or shared resource by the same user, using more than one user name, are not allowed." in err or
                "System error 5" in err):
                return
            logging.warning(err)
            #errcode = process.returncode
        
    def _robust_load_main(self,dir_path):
        main_path = dir_path.load("main.txt")
        try:
            self._net_use(main_path)
        except Exception, e:
            logging.info("ignoring exception when doing net use on {0}, {1}".format(main_path,e.message))
        if os.path.exists(main_path):
            return main_path
        logging.info("'{0}' does not exist, so looking for other copies.".format(main_path))
        for copy_file in dir_path._simple_walk():
            if not self.copy_pattern.match(copy_file):
                continue
            copy_path = dir_path.load(copy_file)
            try:
                self._net_use(copy_path)
            except Exception, e:
                logging.info("ignoring exception when doing net use on {0}, {1}".format(copy_path,e.message))
            if os.path.exists(copy_path):
                logging.info("Doing a copy to main repair. Not sure if this is safe")
                dir_path._simple_remove("main.txt")
                dir_path.save("main.txt",copy_path)
                dir_path._simple_remove(copy_file)
                return copy_path
        raise Exception("The main copy of the file can't be found and no replacement is available. '{0}'".format(main_path))

    def _remove_local_if_any(self,path):
        if path is None:
            unique_name, root = self.local_lambda()
            if os.path.exists(root):
                shutil.rmtree(root)   
        else:
            directory, simple_file_name = self._split(path)
            unique_name, root = directory.local_lambda()
            local_path = root + "/" + simple_file_name
            if os.path.exists(local_path):
                logging.info("Removing '{0}' to save space".format(local_path))
                os.remove(local_path)

    @contextmanager
    def _simple_open_write(self,simple_file_name,size=0,updater=None):
                
        logging.info("open_write('{0}',size={1})".format(simple_file_name,size))
        #Register the file name in the directory
        dir_path=self.directory.join(simple_file_name)
        assert not dir_path._simple_file_exists("main.txt"), "Can't open a file for write if it already exists."
        assert not self._at_least_one(dir_path._simple_walk()), "Can't open a file for write if a directory with files already has the same name ({0},{1})".format(self,simple_file_name)
        unique_name, root = self.local_lambda()
        local_path = root + "/" + simple_file_name
        #Anything in storage (file or directory) can be cleaned up.
        self._create_directory(local_path)
        if os.path.exists(local_path):
            logging.info("File with the same name found locally, but without a directory entry, so removing it. '{0}'".format(local_path))
            os.remove(local_path)
        assert psutil.disk_usage(os.path.dirname(local_path)).free - size > self.leave_space, "Not enough space for '{0}. Need to make space".format(local_path) 

        yield local_path

        logging.info("close('{0}')".format(simple_file_name))
        assert os.path.exists(local_path), "Expect file at '{0}'".format(local_path)
        assert psutil.disk_usage(os.path.dirname(local_path)).free > self.leave_space, "Not enough space for '{0}'. Need to make space".format(path)

        dir_path.save("main.txt",local_path)


    def _remove_internal(self, path):
        for storage_item in self.directory.walk(path):
            storage_path = self.directory.load(storage_item)
            if storage_path != "":  #Dibs files don't point anywhere, so we don't need to delete the file they point to.
                try:
                    logging.debug("/tos.remove('{0}')".format(storage_path))
                    os.remove(storage_path) #In storage, we remove the files, but not the folder
                except:
                    logging.debug("Can't remove (because it's not there): '{0}'".format(storage_path))
        self.directory.rmtree(path)

    def _simple_rmtree(self,log_writer=None):
        logging.info("rmtree -- {0}".format(self.directory))
        self._remove_internal(None)

    def _simple_remove(self,simple_file_name,log_writer=None):
        assert self._simple_file_exists(simple_file_name), "Expect file to exist (and be a file)"
        self._remove_internal(simple_file_name)

    def _simple_getmtime(self,simple_file_name):
        return self.directory._simple_getmtime(simple_file_name + "/main.txt")

    def _simple_join(self,path):
        def closure():
            unique_name, root = self.local_lambda()
            return unique_name, root+"/"+path
        fs = FileShare(self.directory._simple_join(path), local_lambda=closure,leave_space=self.leave_space)
        return fs

    def _simple_walk(self):
        for storage_item in self.directory._simple_walk():
            head, tail = os.path.split(storage_item)
            if tail == "main.txt":
                yield head


class LocalCache(FileCache):
    def __init__(self,directory="local_cache"):
        super(LocalCache, self).__init__()
        self.directory =  os.path.normpath(directory).replace('\\','/')
        if os.path.exists(self.directory): assert not os.path.isfile(self.directory), "A directory cannot exist where a file already exists."

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.directory)

    @property
    def name(self):
        return self.directory

    def _simple_file_exists(self,simple_file_name):
        full_file = self.directory + "/" + simple_file_name
        return os.path.exists(full_file) and os.path.isfile(full_file)

    @contextmanager
    def _simple_open_read(self,simple_file_name,updater=None):
        logging.info("open_read('{0}')".format(simple_file_name))
        file_name = self.directory + "/" + simple_file_name
        assert os.path.exists(file_name), "File doesn't exist in LocalCache. File is '{0}'".format(file_name)

        yield file_name

        logging.info("close('{0}')".format(simple_file_name))

    @contextmanager
    def _simple_open_write(self,simple_file_name,size=0,updater=None):
        logging.info("open_write('{0}',size={1})".format(simple_file_name,size))

        #Register the file name in the directory
        file_name=self.directory + "/" + simple_file_name
        if os.path.exists(file_name): #This is only OK, if it is a directory containing no files (and thus doesn't exist)
            assert not os.path.isfile(file_name), "Can't open a file for write if it already exists."
            assert not self._at_least_one(self.walk(simple_file_name)), "Can't open a file for write if a directory with files already has the same name ({0},{1})".format(self,simple_file_name)
            shutil.rmtree(file_name)
        else:
            pstutil.create_directory_if_necessary(file_name,isfile=True)

        yield file_name

        logging.info("close('{0}')".format(simple_file_name))
        assert os.path.exists(file_name), "File doesn't exist in LocalCache. File is '{0}'".format(file_name)

    def _simple_rmtree(self,log_writer=None):
        logging.info("rmtree -- {0}".format(self.directory))
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)

    def _simple_remove(self,simple_file_name, log_writer=None):
        full_file = self.directory + "/" + simple_file_name
        assert os.path.exists(full_file) and os.path.isfile(full_file), "Expect file to exist (and be a file)"
        os.remove(full_file) #We don't actually remove the folders from the disk, but to this API the folder is gone.

    def _simple_getmtime(self,simple_file_name):
        full_file = self.directory + "/" + simple_file_name
        assert os.path.exists(full_file) and os.path.isfile(full_file), "Expect file to exist (and be a file)"
        return os.path.getmtime(full_file)

    def _simple_join(self,path):
        directory = self.directory + "/" + path
        if os.path.exists(directory): assert not os.path.isfile(directory), "Can't treat an existing file as a directory"
        return LocalCache(directory)

            
            

    def _simple_walk(self):
        for root, dirs, files in os.walk(self.directory):
            root = os.path.normpath(root).replace('\\','/')
            rel = os.path.relpath(root,self.directory).replace('\\','/')
            for file in files:
                yield file if rel == "." else rel + "/" + file


 