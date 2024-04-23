exactUpdate = False
logdelta = np.log(16)
topKbyLinReg = 16

if topKbyLinReg != None:
    outFile = "examples/ScanOSP.exact%d.nSnps%d.logdelta%.2f.out.txt" % (
        exactUpdate,
        topKbyLinReg,
        logdelta,
    )
else:
    outFile = "examples/ScanOSP.exact%d.all.logdelta%.2f.out.txt" % (
        exactUpdate,
        logdelta,
    )

obj = ScanOSP(
    phenoFile="examples/toydata.phe",
    bedFile="examples/toydata",
    windowByPosition=100,
    logdelta=logdelta,
    topKbyLinReg=topKbyLinReg,
    exactUpdate=exactUpdate,
    # prefilterNumSnps = 1000,
    # extractSim = 'examples/LmmGWAS.snps.txt',
    nFolds=5,
    nCV=3,
    outFile=outFile,
    doPlots=True,
    doDebugging=True,
)
