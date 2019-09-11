import os
import pysnptools.util as pstutil

#!!!cmk update everyting and confirm testing
class FileCache(object):
    '''
    A FileCache is class such as :class:`.LocalCache` or :class:`.PeerToPeer` that
       * Copies a (possibly remote) file to a local disk for reading. (If the local file already exists 
         and is up-to-date, retrieval is typically skipped.)
       * Copies a newly-written local file to (possibly remote) storage.
       * Provides an assortment of file- and directory-related commands.

    :Example: #!!!cmk be sure these are all read

        * Peer-to-peer storage - Suppose all machines in a cluster can read & write the storage at 'peertopeer1/common'.
          Also, suppose that 'peertopeer1/192.168.1.105' is readable by all but stored locally
          on machine 192.168.1.105 (and so on for all the machines and their IPs).
        
        >>> from fastlmm.util.filecache import PeerToPeer, ip_address
        >>>
        >>> def id_and_path_function():
        ...     ip = ip_address()
        ...     return ip, 'peertopeer1/{0}'.format(ip)
        >>>
        >>> file_cache = PeerToPeer(common_directory='peertopeer1/common',id_and_path_function=id_and_path_function)
        >>> file_cache
        PeerToPeer('peertopeer1/common',id_and_path_function=...')


        * Remove anything already in remote storage

        >>> file_cache.rmtree()

        * Create a random SNP file and store it remotely

        >>> from pysnptools.snpreader import SnpGen, Dense
        >>> snp_gen = SnpGen(seed=123,iid_count=1000,sid_count=5000)
        >>> with file_cache.open_write('r123.1000x5000.dense.txt') as local_filename:
        >>>     Dense.write(local_filename,snp_gen.read())
        >>> list(file_cache.walk())
        ['r123.1000x5000.dense.txt']

        * Copy back from remote storage (if needed) and then read SNP data from local file.

        >>> with file_cache.open_read('r123.1000x5000.dense.txt') as local_filename:
        >>>    dense = Dense(local_filename)
        >>> print dense[:3,:3].read().val #Read 1st 3 individuals and SNPs
        [[ 0.  0. nan]
         [ 0.  0.  0.]
         [ 0.  0.  0.]]


    Given the appropriate module, FileCache library provides a simple, unified way work to work with any remote storage scheme. It differs from
    virtual file systems, such as CIFS VFS, because:

        * FileCache works with all operating systems and requires no operating system changes.
        * FileCache can take advantage of the highest performance file retrieval methods (e.g. kernel space, peer-to-peer, tree copies, etc).
        * FileCache can take advantage of the highest performance local read and write storage (e.g. SSDs)
        * FileCache can work on top of CIFS VFS and any other remote storage system with an appropriate module.

        The downside of FileCache is that:

        * It generally requires two write-related statements (e.g. 'open_write','Dense.write') instead of just one. Likewise,
          it generally requires to read-related statements (e.g. 'open_read', 'dense[:3,:3].read()' instead of just one.


    Methods & Properties:

        Every FileCache, such as :class:`.LocalCache` and :class:`.PeerToPeer`, has this property:
        :attr:`name`, 
        and these methods:
        :meth:`file_exists`, :meth:`getmtime`, :meth:`join`, :meth:`load`, :meth:`open_read`, :meth:`open_write`, :meth:`remove`, :meth:`rmtree',
        :meth:`save`, and :meth:`walk`. See below for details.

        Details of Methods & Properties:

    '''

    def __init__(self):
        super(FileCache, self).__init__()

    def join(self,path):
        '''
        The :class:`FileCache` created by appending a path to the current :class:`FileCache`.

        :param path: path to join to current :class:`FileCache`
        :type path: string

        :rtype: :class:`FileCache`

        '''
        head = self._normpath(path)
        if head is None:
            return self
        return self._simple_join(head)

    def walk(self,path=None):
        '''
        Generates the relative paths of the files in the :class:`FileCache`. It is OK if there are no files.

        :param path: (Default, None, the current :class:`FileCache`). Optional path (subdirectory, not file) to start in.
        :type path: string

        :rtype: a generator of strings

        :Example:

        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> list(file_cache.walk())
        []

        >>> file_cache.save('file1.txt','Hello')
        >>> list(file_cache.walk())
        ['file1.txt']

        >>> list(file_cache.walk('no_such_sub_path'))
        []
        '''
        for item in self.join(path)._simple_walk():
            yield item if path is None else path + "/" + item
            

    def rmtree(self,path=None,log_writer=None):
        '''
        Delete all files this under this :class:`FileCache`. It is OK if there are no files.

        :param path: (Default, None, the current :class:`FileCache`). Optional a path (subdirectory, not file) to start in.
        :type path: string

        :param log_writer: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_.
        :type log_writer: function or lambda
        '''
        self.join(path)._simple_rmtree(log_writer=log_writer)

    def file_exists(self,file_name): #!!!cmk file_name or filename?
        '''
        Tells if there a file with this name. (A directory with the name doesn't count.)

        :param file_name: The name of the file to look for
        :type file_name: string

        :rtype: boolean

        :Example:

        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> file_cache.file_exists('sub1/file1.txt')
        False

        >>> file_cache.save('sub1/file1.txt','Hello')
        >>> file_cache.file_exists('sub1/file1.txt')
        True
        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_file_exists(simple_file)

    def open_read(self,file_name,updater=None): #!!!cmk how is updater different that log_writer (likewise log_in_place vs. _progress_reporter)
        '''
        Used with a 'with' statement to produce a local copy of a (possibly remote) file.

        :param file_name: The name of the (possibly remote) file to read
        :type file_name: string

        :param updater: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_. !!!cmk good example?
        :type updater: function or lambda

        :rtype: a local filename to read.

        :Example:

        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> file_cache.save('file1.txt','Hello')
        >>> with file_cache.open_read('file1.txt') as local_filename:
        >>>     with open(local_filename,'r') as fp:
        >>>     line = fp.readline()
        >>> line
        'Hello'

        '''
        directory, simple_file_name = self._split(file_name)
        return directory._simple_open_read(simple_file_name,updater=updater)

    def open_write(self,file_name,size=0,updater=None):
        '''
        Used with a 'with' statement to produce a local file name that will be copied to remote storage
        when the 'with' statement is exited.

        :param file_name: The name of the (possibly remote) file to which to write.
        :type file_name: string

        :param size: (default 0) if given, an error will be thrown immediately if local storage doesn't
            have room for that many bytes.
        :type size: number

        :param updater: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_. !!!cmk good example?
        :type updater: function or lambda

        :rtype: a local filename to read.

        :Example:

        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> with file_cache.open_write('file1.txt') as local_filename:
        >>>     with open(local_filename,'w') as fp:
        >>>         fp.write('Hello')
        >>> file_cache.load('file1.txt')
        'Hello'

        '''
        directory, simple_file_name = self._split(file_name)
        return directory._simple_open_write(simple_file_name,size=size,updater=updater)


    def remove(self,file_name,log_writer=None):
        '''
        Remove a file from storage. It is an error to remove a directory this way.

        :param file_name: The name of the (possibly remote) file to remove.
        :type file_name: string

        :param log_writer: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_.
        :type log_writer: function or lambda

        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_remove(simple_file,log_writer=log_writer)

    def save(self, file_name, contents,size=0,updater=None):
        '''
        Write a string to a file in storage.

        :param file_name: The name of the (possibly remote) file to which to write.
        :type file_name: string

        :param contents: What to write to the file.
        :type contents: string

        :param size: (default 0) if given, an error will be thrown immediately if local storage doesn't
            have room for that many bytes.
        :type size: number

        :param updater: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_. !!!cmk good example?
        :type updater: function or lambda


        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> file_cache.save('file1.txt','Hello')
        >>> file_cache.load('file1.txt')
        'Hello'

        '''
        with self.open_write(file_name,size=size,updater=updater) as local_file_name:
            with open(local_file_name,"w") as fp:
                fp.write(contents)

    def load(self, file_name,updater=None):
        '''
        Returns the contents of a file in storage as a string.

        :param file_name: The name of the (possibly remote) file to read
        :type file_name: string

        :param updater: (Default, None). Optional function to which status messages may be written. For example, 
            `log_in_place <http://microsoftgenomics.github.io/PySnpTools/#module-pysnptools.util>`_. !!!cmk good example?
        :type updater: function or lambda

        :rtype: string - what was written in the file.

        >>> from fastlmm.util.filecache import LocalCache
        >>> file_cache = LocalCache('localcache1')
        >>> file_cache.rmtree()
        >>> file_cache.save('file1.txt','Hello')
        >>> file_cache.load('file1.txt')
        'Hello'

        '''
        with self.open_read(file_name,updater=updater) as local_file_name:
            with open(local_file_name,"r") as fp:
                line = fp.readline()
        return line

    def getmtime(self,file_name):
        '''
        Return the modified date of the file in storage.

        :param file_name: The name of the (possibly remote) file of interest
        :type file_name: string

        :rtype: number, the number of seconds since the epoch

        '''
        directory, simple_file = self._split(file_name)
        return directory._simple_getmtime(simple_file)



    @property
    def name(self):#!!!cmk what is the point of this? It seems to have something todo with making nice map_reduce names, but can it be removed?
        '''
        A nice name of this FileCache.

        :rtype: string
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
        if os.name == 'nt':
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
