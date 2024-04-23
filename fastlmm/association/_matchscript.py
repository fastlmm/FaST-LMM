# import os
# import subprocess, sys, os.path
# import numpy as np

# def should_output(bigrow, littlefields):
#    bigrs = bigrow[1]
#    bigchr = bigrow[0]
#    bigdist = float(bigrow[2])
#    for littlerow in littlefields:
#        littlers = littlerow[1]
#        if bigrs == littlers:
#            return False
#        littlechr = littlerow[0]
#        if bigchr == littlechr:
#            littledist = float(littlerow[2])
#            if abs(bigdist - littledist) < 2.0:
#                return False
#    return True


# if __name__ == "__main__":
#    assert len(sys.argv) == 4, "expect 3 arguments"
#    bigfile = sys.argv[1]
#    littlefile = sys.argv[2]
#    outfile = sys.argv[3]

#    logging.info("Loading bim file {0}\n".format(bigfile))
#    bigfields = np.loadtxt(bigfile,delimiter = '\t',dtype = 'str',usecols = (0,1,2,3),comments=None)

#    logging.info("Loading bim file {0}\n".format(littlefile))
#    littlefields = np.loadtxt(littlefile,delimiter = '\t',dtype = 'str',usecols = (0,1,2,3),comments=None)

#    logging.info("comparing\n")
#    import fastlmm.util.util as ut
#    ut.create_directory_if_necessary(outfile)
#    with open(outfile,"w") as out_fp:
#        for bigrow in bigfields:
#            if should_output(bigrow,littlefields):
#                out_fp.write(bigrow[1] + "\n")
