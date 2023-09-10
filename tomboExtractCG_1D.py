import mappy
import os
import difflib
import tensorflow as tf
import numpy as np
import h5py
import math
import re
from tqdm import tqdm
import argparse
def sigExceed(nomsig,siglen):
    stdSig=nomsig[0:3]
    endSig=nomsig[-3:]
    midSig=nomsig[3:-3]
    ratio=len(midSig)//(siglen-6)
    mod=len(midSig)%(siglen-6)
    strP=0
    newmidSig=[]
    for i in range(siglen-6):
        if i<mod:
            newSig=np.mean(midSig[strP:strP+ratio+1])
            newmidSig.append(newSig)
            strP=strP+ratio+1
        else:
            newSig=np.mean(midSig[strP:strP+ratio])
            newmidSig.append(newSig)
            strP=strP+ratio
    newNomSig=np.hstack((stdSig,newmidSig,endSig))
    return newNomSig
def getRawSig(fast5File,stdDna,klength):
    #print(fast5File)
    mersigleng=30#每个聚体的信号长度
    hdf = h5py.File(fast5File, 'r')
    Rds = hdf['Raw/Reads']
    c=[]
    Rds.visit(c.append)
    sig=[]
    readId=str(hdf['Raw/Reads/'][c[0]].attrs['read_id'])
    readId=readId[2:-1]
    for col in hdf['Raw/Reads/'][c[0]]['Signal'][()]:
        sig.append(int(col))
    sigList=[]
    sumList=[]
    try:
        scale=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template'].attrs['scale']
        shift=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template'].attrs['shift']

        chrom=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Alignment'].attrs['mapped_chrom']
        mapstr=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Alignment'].attrs['mapped_start']
        maped=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Alignment'].attrs['mapped_end']
        strand=hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Alignment'].attrs['mapped_strand']
        #print(scale,'::',shift)
        Events = hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Events'][()]
        startPos= hdf['/Analyses/RawGenomeCorrected_001/BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
        #print(Events)
    except:
        return None
    else:
        #获得reference对齐序列
        dnaStr=''
        for base in Events:
            dnaStr=dnaStr+str(base[4])[2]
        #print(mapstr)
        if strand=='+':
            #匹配dnaStr
            list=[i.start() for i in re.finditer(stdDna,dnaStr)]
            for i in list:
                if klength%2==0:
                    avg=klength//2
                    if i>=(avg-1) and (i+avg+1)<len(dnaStr):
                        #信号
                        dnaSeq=str(dnaStr[i-avg+1:i+avg+1])
                        sigStr=readId+'\t'+chrom+'\t+\t'+str(int(mapstr)+i+1)+'\t'+dnaSeq+'\t'
                        startnum=i-avg+1
                        stN=Events[startnum][2]+startPos
                        edN=Events[startnum+klength][2]+startPos
                        siglength=edN-stN
                        if siglength>klength*mersigleng:
                            whRawSig=[]
                            for k in range(klength):
                                stNum=Events[startnum+k][2]+startPos
                                edNum=Events[startnum+k+1][2]+startPos
                                rawSig=sig[stNum:edNum]
                                if len(rawSig)>mersigleng:
                                    rawSig=sigExceed(rawSig,mersigleng)
                                normSig=(rawSig-shift)/scale
                                whRawSig.append(normSig)
                            for x in range(len(whRawSig)):
                                for y in range(len(whRawSig[x])):
                                    sigStr=sigStr+str(whRawSig[x][y])+','
                            #print(sigStr[:-1]) 
                        else:
                            rawSig=sig[stN:edN]
                            #print(rawSig)
                            normSig=(rawSig-shift)/scale
                            for j in range(len(normSig)):
                                sigStr=sigStr+str(normSig[j])+','
                        #print(sigStr[:-1])       
                        sigList.append(sigStr[:-1])
                else:
                    #11 5 -5 +6 5
                    avg=klength//2
                    if i>=(avg) and (i+avg+1)<len(dnaStr):
                        #信号
                        dnaSeq=str(dnaStr[i-avg:i+avg+1])
                        sigStr=readId+'\t'+chrom+'\t+\t'+str(int(mapstr)+i+1)+'\t'+dnaSeq+'\t'
                        startnum=i-avg
                        stN=Events[startnum][2]+startPos
                        edN=Events[startnum+klength][2]+startPos
                        siglength=edN-stN
                        if siglength>klength*mersigleng:
                            whRawSig=[]
                            for k in range(klength):
                                stNum=Events[startnum+k][2]+startPos
                                edNum=Events[startnum+k+1][2]+startPos
                                rawSig=sig[stNum:edNum]
                                if len(rawSig)>mersigleng:
                                    rawSig=sigExceed(rawSig,mersigleng)
                                normSig=(rawSig-shift)/scale
                                whRawSig.append(normSig)
                            for x in range(len(whRawSig)):
                                for y in range(len(whRawSig[x])):
                                    sigStr=sigStr+str(whRawSig[x][y])+','
                            #print(sigStr[:-1]) 
                        else:
                            rawSig=sig[stN:edN]
                            #print(rawSig)
                            normSig=(rawSig-shift)/scale
                            for j in range(len(normSig)):
                                sigStr=sigStr+str(normSig[j])+','
                        #print(sigStr[:-1])       
                        sigList.append(sigStr[:-1])

                    
        else:
            list=[i.start() for i in re.finditer(stdDna,dnaStr)]
            for i in list:
                if klength%2==0:
                    avg=klength//2
                    if i>=(avg-1) and (i+avg+1)<len(dnaStr):
                        #信号
                        dnaSeq=str(dnaStr[i-avg+1:i+avg+1])
                        sigStr=readId+'\t'+chrom+'\t-\t'+str(int(maped)-(i+1))+'\t'+dnaSeq+'\t'
                        startnum=i-avg+1
                        stN=Events[startnum][2]+startPos
                        edN=Events[startnum+klength][2]+startPos
                        siglength=edN-stN
                        if siglength>klength*mersigleng:
                            whRawSig=[]
                            for k in range(klength):
                                stNum=Events[startnum+k][2]+startPos
                                edNum=Events[startnum+k+1][2]+startPos
                                rawSig=sig[stNum:edNum]
                                if len(rawSig)>mersigleng:
                                    rawSig=sigExceed(rawSig,mersigleng)
                                normSig=(rawSig-shift)/scale
                                whRawSig.append(normSig)
                            for x in range(len(whRawSig)):
                                for y in range(len(whRawSig[x])):
                                    sigStr=sigStr+str(whRawSig[x][y])+','
                            #print(sigStr[:-1]) 
                        else:
                            rawSig=sig[stN:edN]
                            #print(rawSig)
                            normSig=(rawSig-shift)/scale
                            for j in range(len(normSig)):
                                sigStr=sigStr+str(normSig[j])+','
                        #print(sigStr[:-1])       
                        sigList.append(sigStr[:-1])
                else:
                    #11 5 -5 +6 5
                    avg=klength//2
                    if i>=(avg) and (i+avg+1)<len(dnaStr):
                        #信号
                        dnaSeq=str(dnaStr[i-avg:i+avg+1])
                        sigStr=readId+'\t'+chrom+'\t-\t'+str(int(maped)-(i+1))+'\t'+dnaSeq+'\t'
                        startnum=i-avg
                        stN=Events[startnum][2]+startPos
                        edN=Events[startnum+klength][2]+startPos
                        siglength=edN-stN
                        if siglength>klength*mersigleng:
                            whRawSig=[]
                            for k in range(klength):
                                stNum=Events[startnum+k][2]+startPos
                                edNum=Events[startnum+k+1][2]+startPos
                                rawSig=sig[stNum:edNum]
                                if len(rawSig)>mersigleng:
                                    rawSig=sigExceed(rawSig,mersigleng)
                                normSig=(rawSig-shift)/scale
                                whRawSig.append(normSig)
                            for x in range(len(whRawSig)):
                                for y in range(len(whRawSig[x])):
                                    sigStr=sigStr+str(whRawSig[x][y])+','
                            #print(sigStr[:-1]) 
                        else:
                            rawSig=sig[stN:edN]
                            #print(rawSig)
                            normSig=(rawSig-shift)/scale
                            for j in range(len(normSig)):
                                sigStr=sigStr+str(normSig[j])+','
                        #print(sigStr[:-1])       
                        sigList.append(sigStr[:-1])

               
    #返回均值化的信号序列，原始信号序列，信号中值和MAD
    return sigList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description='please enter two parameters input direction and output file ...'
    parser.add_argument("-i", "--inputD", help="this is parameter input direction", dest="argI", type=str, default="")
    parser.add_argument("-l", "--inputL", help="this is parameter input length of kmer", dest="argL", type=str, default="")
    parser.add_argument("-o", "--outputF", help="this is parameter output file", dest="argO", type=str, default="")
    args = parser.parse_args()
    if args.argI!='' and args.argO!='' and args.argL!='':        
        path=args.argI
        if not(path.endswith('/')):
            path=path+'/'

        stdDna='CG'
        deepSig_fn=args.argO
        klength=int(args.argL)
        sigList=[]
        for dirpath, dirnames, files in os.walk(path):
            for fast5 in tqdm(files):
            #for fast5 in files:
                if fast5.endswith('.fast5'):
                    deepSigList=getRawSig(path+fast5,stdDna,klength)
                    if deepSigList!=None:
                        for deepSig in deepSigList:
                            if len(deepSig)>11:
                                sigList.append(deepSig)
                                #print(deepSig)  
        if os.path.exists(deepSig_fn):
            os.remove(deepSig_fn)
            os.mknod(deepSig_fn)
            f = open(deepSig_fn, 'w')
            f.writelines([str(line)+'\r\n' for line in sigList])
        else:
            os.mknod(deepSig_fn)
            f = open(deepSig_fn, 'w')
            f.writelines([str(line)+'\r\n' for line in sigList])
    else:
        print('please enter two parameters input direction and output file ...')
        
