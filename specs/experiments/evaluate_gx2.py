"""Reproducible, deterministic gx2 1.5 assessment; no production patches."""
import ast
import json
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gx2
import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, ncx2, norm
from fastlmmclib.quadform import qf


def case(name, x, w, k=None, nc=None, sigma=0.0, oracle=None):
    return dict(name=name, x=x, w=list(w), k=list(k or [1]*len(w)),
                nc=list(nc or [0.0]*len(w)), sigma=sigma, oracle=oracle)


def evaluate(c):
    w=np.array(c['w'], dtype=float)
    k=np.array(c['k'], dtype=np.int32)
    nc=np.array(c['nc'], dtype=float)
    row=dict(c)
    start=time.perf_counter()
    old,fault,trace=qf(c['x'],w,dof=k,noncentrality=nc,sigma=c['sigma'],acc=1e-10)
    row.update(qfc=float(old),qfc_fault=int(fault),qfc_seconds=time.perf_counter()-start)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            p,err,_=gx2.cdf(c['x'],w,k,nc,c['sigma'],0,side='upper',full_output=True)
            row.update(gx2=float(p),gx2_error_output=None if err is None else np.asarray(err).tolist())
        except Exception as error:
            row.update(exception=type(error).__name__+': '+str(error))
    row.update(gx2_seconds=time.perf_counter()-start,warnings=[str(z.message) for z in caught])
    print(json.dumps(row),flush=True)


def main():
    cases=[]
    # Exact elementary / SciPy-distribution oracles independent of QFC.
    for df in [1,2,10]:
        for prob in [0.9,0.05,1e-8,1e-12]:
            x=float(chi2.isf(prob,df))
            cases.append(case(f'chi2_df{df}_p{prob}',x,[1],[df],oracle=prob))
    for nc in [1,20]:
        x=2*float(ncx2.isf(1e-8,3,nc))
        cases.append(case(f'noncentral_{nc}',x,[2],[3],[nc],oracle=1e-8))
    for x in [-10,0,1,10,100]:
        # Sum/difference of two scaled chi2_2 variables = exponentials.
        positive=1.0 if x<0 else 2*math.exp(-x/4)-math.exp(-x/2)
        mixed=(1/3)*math.exp(-x/2) if x>=0 else 1-(2/3)*math.exp(x/4)
        cases.append(case(f'positive_exponentials_{x}',x,[1,2],[2,2],oracle=positive))
        cases.append(case(f'mixed_exponentials_{x}',x,[1,-2],[2,2],oracle=mixed))
    for x in [-2,0,2]:
        cases.append(case(f'zero_weights_{x}',x,[0,0],oracle=float(x<0)))
        cases.append(case(f'normal_{x}',x,[0],[1],sigma=2,oracle=float(norm.sf(x/2))))
        cases.append(case(f'negative_chi2_{x}',x,[-2],[2],oracle=float(chi2.cdf(-x/2,2))))
    cases.append(case('equal_weights',80,[2]*10,[1]*10,oracle=float(chi2.sf(40,10))))
    cases.append(case('zero_padding',10,[0,1,0,2],[2]*4,oracle=2*math.exp(-2.5)-math.exp(-5)))
    # Known mathematical convolution with a tiny second exponential component.
    for small in [1e-3,1e-8,1e-15]:
        for x in [1.,20.]:
            p=(math.exp(-x/2)-small*math.exp(-x/(2*small)))/(1-small)
            cases.append(case(f'condition_{small}_{x}',x,[small,1],[2,2],oracle=p))
    # The exact FaST-LMM model: unit dof, zero noncentrality, positive weights.
    for small in [1e-3, 1e-8]:
        integral, _ = quad(
            lambda theta: math.exp(
                -20 / (2 * (small * math.cos(theta)**2 + math.sin(theta)**2))
            ),
            0, math.pi / 2, epsabs=1e-14, epsrel=1e-12,
        )
        cases.append(case(f'unit_dof_angular_{small}',20,[small,1],
                          oracle=2*integral/math.pi))
        cases.append(case(f'unit_dof_exponential_{small}',20,[small,small,1,1],
                          oracle=math.exp(-10)/(1-small)))
    # Keep tiny negative roundoff eigenvalues; never silently discard them.
    source=ast.parse((Path(__file__).resolve().parents[3] / 'fastlmmclib/tests/test.py').read_text())
    for node in source.body:
        if isinstance(node,ast.FunctionDef) and node.name in ['test_fastlmm_qf','test_sep23_qf']:
            w=ast.literal_eval(node.body[0].value.args[0])
            x=11.756721115473242 if node.name=='test_fastlmm_qf' else 359.4848154869432
            cases.append(case(node.name,x,w))
    rng=np.random.default_rng(20261113)
    for rank in [2,10,50,200]:
        for condition in [1.,1e3,1e8]:
            w=np.geomspace(1/condition,1,rank)*rng.uniform(.95,1.05,rank)
            for z in [0.,3.]:
                x=float(w.sum()+z*np.sqrt(2*np.square(w).sum()))
                cases.append(case(f'psd_rank{rank}_condition{condition}_z{z}',x,w))
    for c in cases:
        evaluate(c)
    # Threads: deterministic basic integration must agree with serial calls.
    w=np.array([.1,.3,.5,.8])
    def run(_):
        return float(gx2.cdf(2,w,np.ones(4),np.zeros(4),0,0,side='upper'))
    serial=[run(i) for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded=list(pool.map(run,range(12)))
    print(json.dumps(dict(name='thread_repeatability',serial=serial,threaded=threaded)),flush=True)


if __name__=='__main__':
    main()
