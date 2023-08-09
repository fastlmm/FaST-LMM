from __future__ import absolute_import
from __future__ import print_function
import sys
import pdb
import os
import gzip
import re
import scipy as SP
from six.moves import range
writesmall = 0

datadir = ""
base = ""
tbi = base+".tbi"
basef=os.path.join(datadir,base)
basefsmall = "chr22.small.txt.gz"
basefsmall = 'chr22.small.txt'
baseftbi = basef + ".tbi"

if writesmall:
    file = gzip.GzipFile(basef, 'r')
    #outfsmall = gzip.GzipFile(basefsmall,'w')
    outfsmall = open(basefsmall,'w')
else:
    file = open(basefsmall,'r')
line = file.readline()
if writesmall:
    outfsmall.write(line)
if line != '##fileformat=VCFv4.1\n':
    print('wrong format')
iline = 0
header = ''
while (line[0:2]=='##'):
    header = header+line
    line = file.readline()
    if writesmall:
        outfsmall.write(line)
    iline+=1;
if line[0:6] == '#CHROM':
    fields = line[1::].strip().split('\t')
    indID=fields[9::]
    fields=fields[0:9]


famfile = open('test.fam','w')
for id in indID:
   outline = '0 ' + id + ' 0 0 0 -9\n'
   famfile.write(outline)
famfile.close()
ninds = len(indID)

line = file.readline()
if writesmall:
    outfsmall.write(line)

bedfile = open('test.bed','wb')
bimfile = open('test.bim','w')

#init bedfile
bedfile.write(b'l\x1b')#bed header
bedfile.write(b'\x01')
#genomap=[]
#geno = []
iline = 0
if writesmall:
    maxline =100
else:
    maxline = 10000000
while (iline < maxline ) and (line !=''):
    vals = line.strip().split('\t')
    if (vals[6]=='PASS'):
        linebim = vals[0] + ' ' + vals[2] + ' 0 ' + vals[1] + ' ' + vals[3] + ' ' + vals[4] + '\n'
        bimfile.write(linebim)
        #genomap.append(vals[0:9])
        #geno.append(vals[9::])

        #write bedfile entries
        nbytes = SP.ceil(ninds/4.0)
        ndone = 0
        for i in range(int(SP.ceil(ninds/4.0))):
            num = min(4,ninds-ndone)
            geno = vals[9+i*4:9+(i)*4+num]
            byte = 0
            #pdb.set_trace()
            for j in range(num):

                if geno[j][0:3]=='0|0':
                    #byte += 0*(2**(2*j))
                    pass
                elif geno[j][0:3]=='0|1' or geno[j][0:3]=='1|0':
                    byte += 2*(2**(2*j))
                elif geno[j][0:3]=='1|1':
                    byte += 3*(2**(2*j))
                else:
                    #pdb.set_trace()
                    byte += 1*(2**(2*j))
                pass

            bytestr = '%c'%byte
            bedfile.write(bytestr)
            ndone +=num
            pass
    line = file.readline()
    if writesmall:
        outfsmall.write(line)
    iline+=1
file.close()
if writesmall:
    outfsmall.close()
bimfile.close()
bedfile.close()

#filetbi = open(baseftbi,'r')
#filetbigz = gzip.GzipFile(baseftbi,'r')
