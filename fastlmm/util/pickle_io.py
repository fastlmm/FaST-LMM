"""
simple module to save and load compressed pickle files
"""

from __future__ import absolute_import
import six.moves.cPickle
import bz2
import sys

 
def save(filename, myobj):
    """
    save object to file using pickle
    
    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """
 
    try:
        f = bz2.BZ2File(filename, 'wb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return
 
    six.moves.cPickle.dump(myobj, f, protocol=2)
    f.close()
 
 
 
def load(filename):
    """
    Load from filename using pickle
    
    @param filename: name of file to load from
    @type filename: str
    """
 
    try:
        f = bz2.BZ2File(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return
 
    myobj = six.moves.cPickle.load(f)
    f.close()
    return myobj


