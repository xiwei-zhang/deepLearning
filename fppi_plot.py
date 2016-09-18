import pdb

import numpy as np
import matplotlib.pyplot as plt



ma3a = open("ROC_eophtha_ma3_less.txt","r")
ma3b = open("ROC_eophtha_ma3b.txt","r")
ma4 = open("ROC_eophtha_ma4_up.txt","r")
ma6 = open("ROC_eophtha_ma6.txt","r")

# exd1 = open("ROC_HEIMED9b.txt","r")
# exd0 = open("ROC_eophtha8_circX2_6.txt","r")
# exd2 = open("ROC_brest.txt","r")
# exd3 = open("ROC_eophtha9b.txt","r")
# exd4 = open("ROC_diaret9b.txt","r")
# exd5 = open("ROC_messidor9b.txt","r")

def score(data):
    list_point = [1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0, 8.0]
    list_sens = [0]*7
            
    fppi = data[0]
    sens = data[1]

    for i in range(7):
        f = 0
        if fppi[0]>list_point[i]:
            list_sens[i] = 0
        else:
            for j in range(len(fppi)):
                if f==1:
                    break
                if fppi[j]>=list_point[i]:
                    f = 1
                    p2 = fppi[j]
                    v2 = sens[j]
                    p1 = fppi[j-1]
                    v1 = sens[j-1]
            k = (v2-v1)/(p2-p1)
            c = v1-k*p1
            list_sens[i] = k*list_point[i]+c
    print list_sens
    print "Final score: ", sum(list_sens)/7
    return round(sum(list_sens)/7,3)




def roc_area(data):
    fpr = data[0]
    sens = data[1]

    area = 0
    area += fpr[1]*sens[1]/2

    for i in range(1,len(fpr)-1):
        aa = ((sens[i] + sens[i+1]) * (fpr[i+1] - fpr[i]))/2
        area += aa
        # print i,sens[i],sens[i+1], sens[i+1]-sens[i], fpr[i], fpr[i+1],fpr[i+1]-fpr[i], aa, area
    print area
    return area


def roc_plot(ma,im_total):
    sens = []
    tp = []
    fp = []
    fdr = []  ## false discovery rate
    
    lines = ma.readlines()
    fn0 = float(lines[0].split()[4])
    tp0 = float(lines[0].split()[1])
    p_total = fn0+tp0

    for l in lines:
        s1 = l.split()
        tp.append(float(s1[1]))
        fp.append(float(s1[2]))
    
    for i in range(len(tp)):
        fdr.append(fp[i]/(tp[i]+fp[i]))
        sens.append(tp[i]/float(p_total))
            
    fdr.reverse()
    sens.reverse()

    return [fdr,sens]




def fproc_plot(ma,im_total):
    sens = []
    tp = []
    fp = []
    fppi = []  ## FP per image
    
    lines = ma.readlines()
    fn0 = float(lines[0].split()[4])
    tp0 = float(lines[0].split()[1])
    p_total = fn0+tp0

    for l in lines:
        s1 = l.split()
        tp.append(float(s1[1]))
        fp.append(float(s1[2]))
    
    for i in range(len(tp)):
        fppi.append(fp[i]/im_total)
        sens.append(tp[i]/float(p_total))
            
    fppi.reverse()
    sens.reverse()

    return [fppi,sens]

im_total_eophtha = 148

roc3a = fproc_plot(ma3a,im_total_eophtha)
roc3b = fproc_plot(ma3b,im_total_eophtha)
roc4 = fproc_plot(ma4,im_total_eophtha)
roc6 = fproc_plot(ma6,im_total_eophtha)


score(roc4)
# roc0 = roc_plot(ma0,im_total_eophtha)

# roc_area(roc1)
plt.plot(roc3a[0],roc3a[1],'-b',label='intensity & geometry: ' + str(score(roc3a)),linewidth=1.0)
plt.plot(roc3b[0],roc3b[1],'-g',label='contextual features: '+str(score(roc3b)),linewidth=1.0)
plt.plot(roc4[0],roc4[1],'-r',label='max-tree: '+str(score(roc4)),linewidth=1.0)
plt.plot(roc6[0],roc6[1],'-k',label='max-tree: '+str(score(roc6)),linewidth=1.0)

plt.legend(loc=4,fontsize='medium')
plt.ylim([0,1])
plt.xlim([0.1,30])
# plt.xlim([0,1])
plt.yticks( [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.xscale('log')

plt.xlabel("Average number of FP/image")
plt.ylabel("Sensitivity")

plt.grid()
plt.show()

pdb.set_trace()
