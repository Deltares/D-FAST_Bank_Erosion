# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:59:54 2022

@author: deijl_ee
"""
import math
import matplotlib.pyplot as plt
import numpy as np

#%%
h=np.arange(2,18,2) # m water depth
Ts = 1.2 # m the draught of the ships
vs = 6 # m/s  relative velocity
s0 = 150 # m
s1 = s0-50
s = np.arange(5,165,5)
a1 = 0.28* Ts**1.25 # for  RHK ship / Motorship


f = np.zeros(len(s))
H0 = np.zeros((len(h),len(s)))
for j in np.arange(len(h)):
    Fr = vs/math.sqrt(9.81*h[j])
    for i in np.arange(len(s)):
        if s[i]>0 and s[i]<s1:
            f = 1
        elif s[i]>=s1 and s[i]<s0:
            f = 0.5 * (1.0 + math.cos((s[i]-s1)/(s0-s1) * math.pi))
        else:
            f = 0
        H0[j,i] = (a1*h[j]*(s[i]/h[j])**(-1/3)*Fr**4)*f
        
#%%
plt.figure()
for i in np.arange(len(h)):
    plt.plot(s,H0[i],label=('h = {} m').format(h[i]))

plt.xlabel('Distance from fairway (m}')
plt.ylabel("$H_0 (m)$")
plt.legend()
plt.ylim(0,2)
plt.xlim(0,160)
plt.grid('major')
plt.savefig('fig5.3.svg')
plt.savefig('fig5.3.png')
plt.savefig('fig5.3.pdf')
          

