import os

#!!!cmk update everyting and confirm testing
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


