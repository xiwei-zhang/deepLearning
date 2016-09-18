import pdb
import os
import re
import math
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from morphee import *
import MorpheeFastPathOpeningPython as FPO
import basicOperations



#######################################
## Feature analysis
def feature_imp(imp):
    # charact_list1= ["vIPic","vTHPic","W","H","area","volume","geodesic length","Npixels above  C","IC","THC","Imean","THmean","Imedian","THmedian","Inner IC","Inner Imean","Inner Imedian","Inner THC","Inner THmean","Inner THmedian","InnerIabovePic","Inner THabovePic","outter IC","outter Imean","outter Imedian","outter THC","outter THmean","outter THmedian","outter","outter THabovePic","circ1","circ2","perimeter"]
    # charact_list1= ["vTHPic","W","H","area","geodesic length","THmedian","N_CC_area1","N_CC_area2","N_CC_area1_L","N_CC_area2_L","circ1"]
    charact_list1= ["minTH","maxTH","meanTH","medianTH","minI","maxI","meanI","medianI","area","height","volume", "pics","candClass","borderDepth","VAR","UO","saturation","perimeter","distCenter","distMin","nPic_h","nPic_l","area_h","area_l"]

    impD = {}
    num = [i for i in range(len(imp))]
    orderV = []
    orderP = []

    while (len(imp)!=0):
        maxV = 0
        maxP = 0
        p = 0
        for i in range(len(imp)):
            if imp[i]>=maxV:
                maxV = imp[i]
                maxP = num[i]
                p = i
    ##    print maxP,":",maxV
        orderP.append(maxP)
        orderV.append(maxV)
        del imp[p]
        del num[p]

    for i in range(24):
        print charact_list1[orderP[i]],orderV[i]
        # if orderP[i] == 0:
        #     print charact_list1[0],0,orderV[i],orderP[i]
        # else:
        #     res = (orderP[i]-1)%20
        #     print charact_list1[(orderP[i]-res-1)/20+1],res+1,orderV[i],orderP[i]
    pdb.set_trace()
    return orderP


#######################################


#### 
# LL = 20
LL = 117
# trSet = 40
####

path = "/home/zhang/work/image/ma6"
# path = "/home/zhang/work/image/Base_Crihan_ma4.0_crossV_500_trees"
# path = "/home/zhang/work/image/Base_Crihan_ma_bea7.0_surEch"
# path = "/home/zhang/work/image/exudate9b"
# path = "/home/zhang/work/bea/ma_result_1.0"
## path = "E:\\work\\image\\Base_Crihan_ma1.0"
# path = "E:\\work\\image\\exudate2"

files = basicOperations.getFiles(path,'txt')


feature_name = [\
        "maxRes","meanRes","meanVessel","inVessel","WH",\
        "area","length","tortousity","circularity",\
        "n_winS_thL_ccS","n_winS_thL_ccL","n_winL_thL_ccS","n_winL_thL_ccL",\
        "n_winS_thH_ccS","n_winS_thH_ccL","n_winH_thL_ccS","n_winH_thL_ccL",]

feature_select = [\
        1,1,1,1,1,\
        1,1,1,1,\
        1,1,1,1,1,1,1,1]
idx = []
for i in range(len(feature_select)):
    if feature_select[i]:
        idx.append(i)

##########################################################################
##########################################################################
## 7/10 new
## I. Load all datasets
learn_set = []
learn_class = []
learn_orig = []

num_candi = []
j=0
for maList in files:

    print "Load:",j,maList
    N_error = 0

    f = open(maList,'r')
    lines = f.readlines()
    rows = len(lines)
    L = 0
    if rows>0:
        L = len(lines[0].split())

    #########
    ## if no candidate
    if rows==0 or L==0:
        m = re.search('txt', maList, re.I)
        end = m.start()-1
        for m in re.finditer('/',maList):  ## for linux
        # for m in re.finditer('\\\\',maList):  ## for windows
            start = m.start()+1
        pattern = maList[start:end]
        rep = maList[:start]
        imrep = rep+pattern+'.png'
        imin = fileRead(imrep)
        filename = pattern+'_rec.png'
        fileWrite(imin,filename)
        order = 'mv -f '+ filename + ' ' + rep # linux
        # order = 'move ' + filename + ' ' + rep # windows
        os.system(order)
        j+=1
        num_candi.append(0) 
        continue
    #########

    num_candi.append(len(lines))

    for line in lines:
        words = line.split()
        word = []

        ###########################or
        ## learn set
        i = 2
        while i<LL-1:
            word.append(float(words[i]))
            i+=1

        # word.append(float(words[L-3]))
        # word_select = [word[i] for i in idx]
        word_select = word
        if word[0]==0:
            N_error += 1
            continue

        learn_set.append(word_select)
    
        ## learn class
        learn_class.append(int(words[-1]))
    
        ## orig list (for trace back)
        orig = maList+' '+words[0]+' '+words[1] + ' '+ words[L-1]
        learn_orig.append(orig) 

        ############################
    num_candi[j] -= N_error
    f.close()
    j+=1

## II. selection
test_set = []
test_class = []
test_orig = []
for num in range(len(files)):
    print num
    for x in test_set:
        learn_set.append(x)
    for x in test_class:
        learn_class.append(x)
    for x in test_orig:
        learn_orig.append(x)

    test_set = []
    test_class = []
    test_orig = []

    for i in range(num_candi[num]):
        test_set.append(learn_set[0])
        test_class.append(learn_class[0])
        test_orig.append(learn_orig[0])
        del learn_set[0]
        del learn_class[0]
        del learn_orig[0]


    if num_candi[num]==0:
        continue

	######################################
    ## Sampling the learning set
    if 0:
        co_mult = 3
        learn_setS = []
        learn_classS = []
        learn_origS = []
    
        s = sum(learn_class) * co_mult
        i = 0
        # n = 0
        while i<s:
            n = random.randint(0,(len(learn_class)-1))
            if learn_class[n] == 0:
                learn_setS.append(learn_set[n])
                learn_classS.append(learn_class[n])
                learn_origS.append(learn_orig[n])
                i+=1
            #    n+=1
            # else:
            #     n+=1
    
        i = 0
        while i<len(learn_class):
            if learn_class[i] == 1:
                learn_setS.append(learn_set[i])
                learn_classS.append(learn_class[i])
                learn_origS.append(learn_orig[i])
            i+=1

    #######################################

    #######################################
    ## Multipling the learning set
    if 1:
        learn_setS = learn_set
        learn_classS = learn_class
        learn_origS = learn_orig
    
        s1 = sum(learn_class)
        # s2 = len(learn_class) - s1
        s2 = (len(learn_class) - s1)/3

        i = 0
        # n = 0
        ts = []
        tc = []
        to = []
        while i<len(learn_class):
            if learn_class[i]==1:
                ts.append(learn_set[i])
                tc.append(learn_class[i])
                to.append(learn_orig[i])
            i+=1
    
        for tt in range(s2/s1):
            learn_setS = learn_setS + ts
            learn_classS = learn_classS + tc
            learn_origS = learn_origS + to 
#        i = 0
#        while i<len(learn_class):
#            if learn_class[i] == 1:
#                learn_setS.append(learn_set[i])
#                learn_classS.append(learn_class[i])
#                learn_origS.append(learn_orig[i])
#            i+=1
    #######################################

    
    #######################################
    ## I. learning
    
    rf = RandomForestClassifier(n_estimators = 200,  n_jobs = 6) 

    rf.fit(learn_setS,learn_classS)
#    lr = LogisticRegression()
#    lr.fit(learn_setS,learn_classS)

    # from sklearn.svm import SVC
    # from sklearn.feature_selection import RFE

    # svc = SVC(kernel="linear", C=1)
    # rfe = RFE(estimator = svc, n_features_to_select=1, step=1)
    # aaa = np.asarray(learn_setS)
    # rfe.fit(aaa, learn_classS)
    # pdb.set_trace()
    #######################################


    # imp = rf.feature_importances_.tolist()
    # feature_imp(imp)
    # pdb.set_trace()

    #################################
    ## predict
    # result = rf.predict(test_set)
    result = rf.predict_proba(test_set)
    #################################
    
    #################################
    ## reconstruct
    maList = files[num]
    m = re.search('txt', maList, re.I)
    end = m.start()-1
    for m in re.finditer('/',maList):  ## for linux
    # for m in re.finditer('\\\\',maList):  ## for windows
        start = m.start()+1
    pattern = maList[start:end]
    rep = maList[:start]
    imrep = rep+pattern+'.png'
    imin = fileRead(imrep)
    
    imtemp1 = getSame(imin)
    imtemp2 = getSame(imin)
    imtemp3 = getSame(imin)
    ImSetConstant(imtemp1,0)
    i = 0
    while i<len(test_class):
        # if result[i] == 1:
        test_words = test_orig[i].split()
        x = int(test_words[1])
        y = int(test_words[2])
        offset = imtemp1.offsetFromCoords(x,y,0)
        if imtemp1.getPixel(offset) < int(round(result[i][1]*255)):
            imtemp1.setPixel(offset,int(round(result[i][1]*255)))
        i+=1
        # else:
        #     i+=1
    #################################
    
    ImThreshold(imin,1,255,255,0,imtemp3)
    FPO.ImUnderBuild(imtemp3,imtemp1,8,imtemp2)
    filename = pattern+'_rec.png'
    fileWrite(imtemp2,filename)
    order = 'mv -f '+ filename + ' ' + rep  ## linux
    # order = 'move ' + filename+ ' ' + rep   ## windows
    os.system(order)
    j+=1
   

##########################################################################
##########################################################################
pdb.set_trace()


