from __future__ import absolute_import
import warnings

class OutputWriter(object):
    ''' Centralize output without messing with sys.stdout. '''
    _streams = []

    def addOutputStream(self, out):
        for o in self._streams:
            if o is out:
                return
        self._streams.append(out)
    
    def write(self, msg):
        warnings.warn("Pr and OutputWriter are deprecated. Use the standard logging.info() instead", DeprecationWarning)
        for s in self._streams:
            s.write(msg)
    
    def prin(self, msg):
        warnings.warn("Pr and OutputWriter are deprecated. Use the standard logging.info() instead", DeprecationWarning)
        self.write(msg+'\n')

    def close(self):
        for s in self._streams:
            s.close()

    def fileno(self):
        for s in self._streams:
            if 'fileno' in dir(s):
                return s.fileno()
        return 0

    def flush(self):
        for s in self._streams:
            if 'flush' in dir(s):
                s.flush()

# This is the only OutputWriter instance that you should use.
Pr = OutputWriter()
