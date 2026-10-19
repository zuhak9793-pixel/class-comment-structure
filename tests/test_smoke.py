"""Synthetic inputs only: no downloads, checkpoints or research data."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recovery
from prepare import mapped_norm, Union
import models
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Smoke(unittest.TestCase):
    def test_recovery_offsets_features(self):
        text='Opening words.\n  Do *synthetic* work.\nClosing words.'
        normalized, offsets=mapped_norm(text,'space')
        spans=recovery.spans(recovery.norm('Do synthetic work.','space'),normalized)
        self.assertEqual(len(spans),1)
        s,e=spans[0]; a,b=offsets[s],offsets[e-1]+1
        self.assertEqual(text[a:b],'Do *synthetic* work.')
        features=[a/len(text),(b-a)/len(text),np.log1p(len(text[:a].split())),np.log1p(len(text[b:].split()))]
        self.assertEqual(features,[17/52,20/52,np.log1p(2),np.log1p(2)])

    def test_ambiguous_and_word_boundaries(self):
        self.assertEqual(len(recovery.spans('synthetic','synthetic synthetic')),2)
        self.assertEqual(recovery.spans('work','worker'),[])
        self.assertEqual(recovery.spans('','anything'),[])

    def test_fit_and_scores(self):
        x=np.array([[0.,0.,0.,0.],[1.,1.,1.,1.],[.1,.1,.1,.1],[.9,.9,.9,.9]])
        y=np.array([0,1,0,1])
        z=StandardScaler().fit_transform(x)
        m=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=1000,tol=1e-4,random_state=models.SEED).fit(z,y)
        p=(m.predict_proba(z)[:,1]>=.5).astype(np.uint8)
        ma,mi,_=models.calc(models.counts(y[:,None],p[:,None,None]).sum(0))
        self.assertEqual(float(ma[0]),1.)
        self.assertEqual(float(mi[0]),1.)
        u=Union(['a','b']);u.join('b','a');self.assertEqual(u.find('b'),'a')

if __name__=='__main__': unittest.main()
