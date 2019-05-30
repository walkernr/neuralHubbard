# -*- coding: utf-8 -*-
"""
Created on Wed May 8 11:13:55 2019

@author: Nicholas
"""

import os
import numpy as np
import re
from tqdm import tqdm

NAME = 'hubbard'
CWD = os.getcwd()
DATDIR = CWD+'/data'
SFL = ['F90']
NFL = len(SFL)
DATFLS = [[DATDIR+'/'+SFL[i]+'/'+f for f in os.listdir(DATDIR+'/'+SFL[i]) if '.dat' in f] for i in range(NFL)]
NBETA = 25
NS = 10000
NK = 16
NT = 100


def read_file(fn):
    tfl, beta = np.array(re.findall(r'\d+', fn.split('/')[-1])).astype(np.float)
    dat = []
    n = 0
    with open(fn, 'r') as f:
        it = iter(f)
        for l in it:
            if 'config' in l:
                n += 1
                d = ''
                while True:
                    dl = it.readline()
                    if dl.strip() == '' or 'cpu  ,time=' in dl:
                        break
                    else:
                        d += dl
                dat.append(d.strip().split())
    dat = np.array(dat).astype(np.float32).reshape(n, NT, NK)
    return dat, tfl/1e4, beta

FL = np.zeros((NFL, NBETA), dtype=np.float32)
BETA = np.zeros((NFL, NBETA), dtype=np.float32)
DAT = np.zeros((NFL, NBETA, NS, NT, NK), dtype=np.float)
print('parsing data files')
for i in tqdm(range(NFL)):
    for j in tqdm(range(NBETA)):
        DAT[i, j], FL[i, j], BETA[i, j] = read_file(DATFLS[i][j])
TFL = np.array([SFL[i][1:] for i in range(NFL)]).astype(np.float32)/1e2
TBETA = BETA[0]

np.save(CWD+'/results/%s.%d.%d.tfl.npy' % (NAME, NT, NK), TFL)
np.save(CWD+'/results/%s.%d.%d.tbeta.npy' % (NAME, NT, NK), TBETA)
np.save(CWD+'/results/%s.%d.%d.fl.npy' % (NAME, NT, NK), FL)
np.save(CWD+'/results/%s.%d.%d.beta.npy' % (NAME, NT, NK), BETA)
np.save(CWD+'/results/%s.%d.%d.dat.npy' % (NAME, NT, NK), DAT)