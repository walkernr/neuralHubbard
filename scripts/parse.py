# -*- coding: utf-8 -*-
"""
Created on Wed May 8 11:13:55 2019

@author: Nicholas
"""

import os
import numpy as np
from tqdm import tqdm

NAME = 'hubbard'
CWD = os.getcwd()
DATDIR = CWD+'/data'
DATFLS = [DATDIR+'/'+f for f in os.listdir(DATDIR) if '.dat' in f]
N = len(DATFLS)
NK = 16
NT = 100


def read_file(fn):
    ds = []
    with open(fn, 'r') as f:
        it = iter(f)
        for l in it:
            if 'Results from dca for nl,nwn,cluster,ndim=' in l:
                p0 = it.readline().strip().split()
            if 'warm,skip,meas,ntr,run,niter,nuse,dfac,ifac' in l:
                p1 = it.readline().strip().split()
            if 'random number, isb, isi, isd' in l:
                p2 = it.readline().strip().split()
            if 'ed    ,  tprime,    U   ,   beta  ,   V, tperp' in l:
                p3 = it.readline().strip().split()
            if 'config' in l:
                d = ''
                while True:
                    dl = it.readline()
                    if dl.strip() == '' or 'cpu  ,time=' in dl:
                        break
                    else:
                        d += dl
                ds.append(d.strip().split())
    ed, beta = np.array(p3).astype(np.float)[[0,3]]
    ds = np.array(ds).astype(np.float)
    return ds, ed, beta

ED = np.zeros(N, dtype=np.float)
BETA = np.zeros(N, dtype=np.float)
DS = np.zeros((N, 4160, NK*NT), dtype=np.float)
print('parsing data files')
for i in tqdm(range(N)):
    DS[i], ED[i], BETA[i] = read_file(DATFLS[i])

UED = np.unique(ED)
NED = UED.size
UBETA = np.unique(BETA)
NBETA = UBETA.size

DAT = np.zeros((NED, NBETA, 4160, 100, 16), dtype=np.float)
print('reorganizing data')
for i in tqdm(range(N)):
    j = np.where(ED[i] == UED)[0][0]
    k = np.where(BETA[i] == UBETA)[0][0]
    DAT[j, k, :, :, :] = DS[i, :, :].reshape(4160, 100, 16)

np.save(CWD+'/%s.%d.ed.npy' % (NAME, NK), UED)
np.save(CWD+'/%s.%d.beta.npy' % (NAME, NK), UBETA)
np.save(CWD+'/%s.%d.%d.dat.npy' % (NAME, NK, NT), DAT)