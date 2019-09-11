"""Tools for reading and writing files, locally or across clusters.
"""

def ip_address():
    '''
    !!!cmk doc
    '''
    import socket
    #see http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    return ([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1],[[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]) 


#!!!cmk update everyting and confirm testing
from fastlmm.util.filecache.filecache import FileCache
from fastlmm.util.filecache.localcache import LocalCache
from fastlmm.util.filecache.peertopeer import PeerToPeer
from fastlmm.util.filecache.distributedbed import DistributedBed

 