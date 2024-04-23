exactUpdate = False
topKbyLinReg = 16
logdelta = np.log(16)

if topKbyLinReg is not None:
    outFile = "examples/ScanISP.exact%d.nSnps%d.logdelta%.2f.out.txt" % (
        exactUpdate,
        topKbyLinReg,
        logdelta,
    )
else:
    outFile = "examples/ScanISP.exact%d.all.logdelta%.2f.out.txt" % (
        exactUpdate,
        logdelta,
    )

obj = ScanISP(
    phenoFile="examples/toydataTrain.phe",
    # phenoTestFile = 'examples/toydataTest.phe',
    bedFile="examples/toydata",
    # fastlmmPath = 'C:/Users/t-baraki/Users/t-baraki/adaptivelmm/software/FaSTLMM.205.Win/Cpp_MKL',
    fastlmmPath=r"C:\Users\lippert\Projects\FaST-LMM_old\FastLmm\CPP\x64\MKL-Release",
    windowByPosition=1e1,
    logdelta=logdelta,
    topKbyLinReg=topKbyLinReg,
    exactUpdate=exactUpdate,
    # extractSim = 'examples/LmmGWAS.snps.txt',
    outFile=outFile,
    doPlots=False,
    doDebugging=True,
)
