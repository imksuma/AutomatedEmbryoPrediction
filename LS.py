import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread
from sklearn.metrics import confusion_matrix
from helper import featureExtractionFBP, featureExtractionTP
import numpy as np

from os import listdir
import os.path
import pickle
from Optimization import BBatAlgorithm

class LearningSystemFBP():

    def __init__(self,loadDataSet=False):
        self.RANDOM_STATE = 123
        self.machine = RandomForestClassifier(warm_start=True,
                                              max_features='sqrt',
                                              oob_score=True,
                                              random_state=self.RANDOM_STATE)
        self.oobErrorList=[]
        self.errorTest=[]
        self.fileName="LearningSystemFBP.pkl"
        self.min_estimator = 30
        self.max_estimator = 100
        self.loadDataSet=loadDataSet
        self.selectedFeature=None
        self.number = 0
        self.n_estimator = None
        self.X = None
        self.y = None
        self.Xtest = None
        self.ytest = None

    def save(self):
        with open(self.fileName,'wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.fileName,'rb') as input:
            obj = pickle.load(input)
            self.RANDOM_STATE = obj.RANDOM_STATE
            self.machine = obj.machine
            self.oobErrorList = obj.oobErrorList
            self.errorTest = obj.errorTest
            self.selectedFeature = obj.selectedFeature
            self.n_estimator = obj.n_estimator
            self.X = obj.X
            self.y = obj.y
            self.Xtest = obj.Xtest
            self.ytest = obj.ytest

    def readImageAndExtractImageFeature(self,fname):
        if fname.find("Frame") != -1 and fname.find(".png") != -1:
            img = imread(fname, as_grey=False)
            if img.shape.__len__() == 2:
                return featureExtractionFBP(img)
            else:
                return None
        else:
            return None

    def getDataSet(self, path):
        fullPath = []
        for dir in listdir(path):
            for f in listdir(path+dir):
                fullPath.extend([path+dir+"\\"+f])

        X=[]
        y=[]
        for dir in fullPath:
            print(dir)
            # tempX=[self.featureExtraction(imread(dir+"\\"+imgFname, as_grey=False)) for imgFname in listdir(dir)]
            tempX=[self.readImageAndExtractImageFeature(dir+"\\"+imgFname) for imgFname in listdir(dir)]
            tempX=[ii for ii in tempX if type(ii) != type(None)]
            X.extend(tempX)

            strSplit = dir.split("\\")
            y.extend([strSplit[strSplit.__len__()-1] for i in range(tempX.__len__())])

        return (X,y)

    def trainTemporaryMachine(self,x,X,y,Xtest,ytest):
        selectedFeature = [idx for idx, val in enumerate(x) if val==1]
        clf = RandomForestClassifier(warm_start=True,
                                              max_features='sqrt',
                                              oob_score=True,
                                              random_state=self.RANDOM_STATE)
        clf.set_params(n_estimators=20)
        if np.sum(selectedFeature) == 0: return 10

        # self.machine.fit(X[:,selectedFeature],y)
        clf.fit(X[:,selectedFeature], y)

        yPTest = clf.predict(Xtest[:,selectedFeature])
        yP = clf.predict(X[:,selectedFeature])

        yPTestval = clf.predict_proba(Xtest[:,selectedFeature])
        yPval = clf.predict_proba(X[:,selectedFeature])

        yPTestval = np.max(yPTestval, axis=1)
        yPval = np.max(yPval, axis=1)

        acc1 = np.sum(yPTestval[yPTest==ytest])
        acc2 = np.sum(yPval[yP==y])
        # coorectPred = np.sum(yP==y)+np.sum(yPTest==ytest)
        # for ii in range(ytest.size):
        #     if yPTest[ii] == ytest[ii] : coorectPred += 1
        #
        # for ii in range(y.size):
        #     if yP[ii] == y[ii] : coorectPred += 1
        # acc = coorectPred/(y.size+ytest.size)

        return -(acc1+acc2)

    def train(self):
        if not self.loadDataSet:
            trainDataSetPath = "training\\"
            testDataSetPath = "test\\"

            self.X, self.y = self.getDataSet(trainDataSetPath)
            self.Xtest, self.ytest = self.getDataSet(testDataSetPath)

            X=np.array([np.array(ii) for ii in self.X])
            y=np.array(self.y)

            Xtest=np.array([np.array(ii) for ii in self.Xtest])
            ytest=np.array(self.ytest)
        else :
            print("load dataset")
            self.load()

            X=np.array([np.array(ii) for ii in self.X])
            y=np.array(self.y)

            Xtest=np.array([np.array(ii) for ii in self.Xtest])
            ytest=np.array(self.ytest)
            self.selectedFeature = [1, 2, 3, 4, 5, 6, 8, 17, 19, 20]

        if self.selectedFeature is None:
            self.selectedFeature = [idx for idx in range(X.shape[1])]

            D = X[0].__len__()
            opt = BBatAlgorithm(D=D,
                                NP=40,
                                N_Gen=30,
                                A=0.4,
                                r=0.8,
                                Qmin=0,
                                Qmax=2,
                                function=lambda x: self.trainTemporaryMachine(x,X,y,Xtest,ytest))
            opt.move_bat()
            self.selectedFeature = [idx for idx, val in enumerate(opt.best) if val==1]
        else :
            print("using previous selected feature")
            self.oobErrorList = []
            self.errorTest = []

        X=X[:,self.selectedFeature]
        Xtest=Xtest[:,self.selectedFeature]

        self.machine = RandomForestClassifier(warm_start=True,
                                              max_features='sqrt',
                                              oob_score=True,
                                              random_state=self.RANDOM_STATE)

        tempOObError = 1
        min_estimator = 20#self.min_estimator
        tempParams = min_estimator
        max_estimator = self.max_estimator

        for i in range(min_estimator,max_estimator+1):
            self.machine.set_params(n_estimators=i)
            self.machine.fit(X,y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - self.machine.oob_score_
            self.oobErrorList.append((i,oob_error))

            yPTest = self.machine.predict(Xtest)

            oob_error2 = 1 - np.mean(ytest==yPTest)
            self.errorTest.append((i,oob_error2))

            if oob_error2 < tempOObError:
                tempParams = i
                tempOObError = oob_error2

        self.n_estimator = tempParams

    def build(self,retrain=True):
        if os.path.isfile(self.fileName) and not retrain:
            self.load()
        else:
            self.train()
            self.save()

        X=np.array([np.array(ii) for ii in self.X])
        y=np.array(self.y)

        Xtest=np.array([np.array(ii) for ii in self.Xtest])
        ytest=np.array(self.ytest)

        X=X[:,self.selectedFeature]
        Xtest=Xtest[:,self.selectedFeature]

        print("retrain new machine")
        machine = RandomForestClassifier(warm_start=True,
                                              max_features='sqrt',
                                              oob_score=True,
                                              random_state=self.RANDOM_STATE)

        machine.set_params(n_estimators=self.n_estimator)
        machine.fit(X,y)

        ypred = machine.predict(X)
        ytpred = machine.predict(Xtest)

        ytpred_prob = machine.predict_proba(Xtest)
        print(np.sum(np.max(ytpred_prob,axis=1)[ytpred==ytest]))

        cnf_matrix_train = confusion_matrix(y, ypred)
        print(cnf_matrix_train)

        cnf_matrix_test = confusion_matrix(ytest, ytpred)
        print(cnf_matrix_test)
        print(np.mean(ytest==ytpred))
        print(np.min(self.errorTest))
        print(self.n_estimator)

    def plotOOBError(self):
        xs, ys = zip(*self.oobErrorList)
        plt.plot(xs,ys,label="train")

        xtest, ytest = zip(*self.errorTest)
        plt.plot(xtest, ytest,label="test")

        plt.xlim(self.min_estimator, self.max_estimator)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()

    def predict_proba(self,fname=None,img=None):
        if fname is not None :
            feature = self.readImageAndExtractImageFeature(fname)
        if img is not None :
            feature = featureExtractionFBP(img)
        if feature is None: return None
        # print(self.machine.predict_proba([feature[self.selectedFeature]])[0])
        return self.machine.predict_proba([feature[self.selectedFeature]])[0]

class LearningSystemTP():

    def __init__(self,loadDataSet=False):
        self.RANDOM_STATE=123
        self.machine = LogisticRegressionCV(tol=1e-6)
        self.pathTraining = "training\\"
        self.pathTest = "test\\"
        self.fileName = "LearningSystemTP.pkl"
        self.selectedFeature = None
        self.loadDataSet = loadDataSet
        self.X = None
        self.y = None
        self.Xtest = None
        self.ytest = None
        self.n_estimator = 20
        self.min_estimator = 30
        self.max_estimator = 100

    def save(self):
        with open(self.fileName,'wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.fileName,'rb') as input:
            obj = pickle.load(input)
            self.machine = obj.machine
            self.selectedFeature = obj.selectedFeature
            self.X = obj.X
            self.y = obj.y

            self.Xtest = obj.Xtest
            self.ytest = obj.ytest

            self.n_estimator = obj.n_estimator

    def readAndValidateImg(self, fname):
        if fname.find("Frame") != -1 and fname.find(".png") != -1:
            img = imread(fname, as_grey=False)
            if img.shape.__len__() == 2:
                print(fname)
                return featureExtractionTP(img)
            else:
                return None
        else:
            return None

    def pred_prob(self,img1=None,img2=None):
        if img1 is None and img2 is None:
            print("error img1 and img2 is found to be None")
        if img2 is None:
            print("error img2 should not be None")
        desc1 = None
        if img1 is not None:
            desc1 = featureExtractionTP(img=img1)
        desc2 = featureExtractionTP(img=img2)
        feature = self.extractDiffFromImage(desc1,desc2)
        return self.machine.predict_proba([feature])[0]

    def extractDiffFromImage(self,hedfit0,hedfit1):
        """ method ini memiliki tugas untuk membuat fitur dari dua gambar yg berdampingang
        'contigous frame'."""
        if hedfit0 is None:
            hedfit0 = np.copy(hedfit1)
        return np.sqrt((hedfit0-hedfit1)**2)

    def getDataFromTheClass(self, classPath):
        """ method ini memiliki tugas untuk mengolah seluruh gambar pada
            kelas yg direpresentasikan sebagai sebuah folder dg nama 'classPath'.
            Tugas fungsi ini ada 3, pertama untuk mengembalikan gambar pertama
            pada folder, kedua membuat list 'feature' dari gambar yang berdampingan,
            ketiga untuk mengembalikan gambar terakhir """
        listImg = []
        for imgfname in listdir(classPath):
            listImg.extend([classPath+imgfname])
        f_img = None
        idx = 0
        while f_img is None:
            f_img = self.readAndValidateImg(fname=listImg[idx])
            idx += 1

        tempX = []

        l_img = None
        l_idx = listImg.__len__()
        ff_img = f_img
        while idx < l_idx:
            ss_img = None
            while ss_img is None:
                ss_img = self.readAndValidateImg(listImg[idx])
                if ss_img is None:
                    idx += 1
            idx +=1
            l_img = ss_img

            tempX.extend([self.extractDiffFromImage(ff_img,ss_img)])
            ff_img = ss_img

        return f_img, tempX, l_img

    def getDataSet(self, path):
        fullPath = []
        for sequanceName in listdir(path):
            fullPath.extend([path+sequanceName+"\\"])

        X = []
        y = []

        for sequancePath in fullPath:
            tempDir = []
            for dir in listdir(sequancePath):
                tempDir.extend([sequancePath+dir+"\\"])

            firstImg = None
            for class_of_the_seq in tempDir:
                f_img, tempX, l_img = self.getDataFromTheClass(class_of_the_seq)
                tx = [self.extractDiffFromImage(firstImg,f_img)]
                if firstImg is None:
                    ty = [0]
                else:
                    ty = [1]
                firstImg = l_img
                tx.extend(tempX)
                ty.extend([0 for i in tempX])

                X.extend(tx)
                y.extend(ty)

        return (X,y)

    def train(self):
        if self.loadDataSet:
            self.load()
            X = self.X
            y = self.y

            Xtest = self.Xtest
            ytest = self.ytest

            X=np.array([np.array(ii) for ii in X])
            y=np.array(y)
        else:
            X,y = self.getDataSet(self.pathTraining)
            Xtest,ytest = self.getDataSet(self.pathTest)

            X=np.array([np.array(ii) for ii in X])
            y=np.array(y)

            Xtest=np.array([np.array(ii) for ii in Xtest])
            ytest=np.array(ytest)

            self.X = X
            self.Xtest = Xtest
            self.y = y
            self.ytest = ytest

            self.save()
        self.selectedFeature = [idx for idx in range(X.shape[1])]

        X=X[:,self.selectedFeature]

        self.machine = RandomForestClassifier(warm_start=True,
                                              max_features='sqrt',
                                              oob_score=True,
                                              random_state=self.RANDOM_STATE)

        self.machine.set_params(n_estimators=20)
        self.machine.fit(X,y)

    def build(self,retrain=True):
        if retrain:
            """ train the classifier """
            self.train()
            self.save()
        else:
            """ load the trained model """
            self.load()

        self.report()

    def report(self):
        """ fungsi ini bertugas untuk membuat laporan yang berkaitan dengan
            hasil leraning """
        ys = self.ytest#[0:400]
        xs = np.array(range(ys.__len__()))
        plt.plot(xs,ys,label="label")

        ytest = self.machine.predict_proba(self.Xtest[:,self.selectedFeature])
        xtest = np.copy(xs)

        plt.plot(xtest, ytest[:,1],label="pred")

        plt.xlabel("frame")
        plt.ylabel("pred")
        plt.legend(loc="upper right")
        plt.show()

class LearningSystemCRF():

    def __init__(self):
        self.fbp = LearningSystemFBP()
        self.fbp.load()
        self.tp = LearningSystemTP()
        self.tp.load()
        self.X_fbp = None
        self.y = None
        self.X_tp = None
        self.fileName = "LearningSystemCRF.pkl"

    def save(self):
        with open(self.fileName,'wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.fileName,'rb') as input:
            obj = pickle.load(input)
            self.X_fbp = obj.X_fbp
            self.y = obj.y
            self.X_tp = obj.X_tp

    def makeNewPrediction(self, list_of_cp, pred_prob, tp_prob):
        if list_of_cp is None:
            return [[[None, -np.log2(ii)] for ii in pred_prob]]
        else :
            temp_oc = []
            ll = list_of_cp[-1]
            for idx, ii in enumerate(pred_prob):
                tidx = idx
                ts = -np.log2(ii)+ll[idx][1]
                for idxl in range(idx+1):
                    tn = -np.log2(ii)+ll[idxl][1]
                    if idxl < idx:
                        tn = tn+1-tp_prob[1]
                    else:
                        tn = tn+tp_prob[1]
                    if ts > tn:
                        ts = tn
                        tidx = idxl
                temp_oc.extend([[tidx,ts]])
            list_of_cp.extend([temp_oc])
            return list_of_cp

    def translateListOfCP(self,list_of_cp):
        pred = []
        rr = np.sort(-np.array(range(list_of_cp.__len__())))*-1
        iidx = list_of_cp[-1][-1][0]
        val = list_of_cp[-1][-1][-1]
        for ii in list_of_cp[-1]:
            if ii[1] < val:
                iidx = ii[0]
                val = ii[1]
        pred.extend([iidx])
        for ii in rr:
            iidx = list_of_cp[ii][iidx][0]
            pred.extend([iidx])
        # print(pred)
        arrR = np.array(pred)
        print(arrR[rr])
        return arrR[rr]

    def predictPerSequence(self, pathSequance, load=False):
        if not load:
            fullPath = []
            for className in listdir(pathSequance):
                fullPath.extend([pathSequance+"\\"+className])

            X_fbp=[]
            X_tp = []
            y=[]
            ff_img = None
            for dir in fullPath:
                print(dir)
                # tempX=[self.featureExtraction(imread(dir+"\\"+imgFname, as_grey=False)) for imgFname in listdir(dir)]
                tempX=[self.fbp.predict_proba(dir+"\\"+imgFname) for imgFname in listdir(dir)]
                tempX=[ii for ii in tempX if ii is not None]
                X_fbp.extend(tempX)

                strSplit = dir.split("\\")
                y.extend([strSplit[strSplit.__len__()-1] for i in range(tempX.__len__())])

                f_img, tempX, l_img = self.tp.getDataFromTheClass(dir+"\\")
                tx = [self.tp.extractDiffFromImage(ff_img,f_img)]
                ff_img = l_img
                tx.extend(tempX)
                X_tp.extend(tx)
            pred_prob_tp = self.tp.machine.predict_proba(X_tp)

            self.X_fbp = X_fbp
            self.y = y
            self.X_tp = X_tp
            # print(X_fbp)
            # print(pred_prob_tp)
            self.save()
        else:
            self.load()
            X_fbp = self.X_fbp
            y = self.y
            X_tp = self.X_tp
            pred_prob_tp = self.tp.machine.predict_proba(X_tp)
        X_fbp = np.array(X_fbp)
        X_fbp[X_fbp==0] = 1e-62
        list_of_cp = None
        for idx, ii in enumerate(X_fbp):
            list_of_cp = self.makeNewPrediction(list_of_cp,ii,pred_prob_tp[idx,:])
        # for ii in list_of_cp: print(ii)
        arrR = np.array(self.translateListOfCP(list_of_cp))+1
        return arrR

    def testPredSeq(self):
        path = "test\\E03"
        print(path)
        pp = self.predictPerSequence(path,load=False)
        print(pp)
        print(self.y.__len__())
        cnf_matrix_train = confusion_matrix(self.y, [ii.__str__() for ii in pp])
        print(np.mean([int(ii) for ii in self.y]==pp))#[ii.__str__() for ii in pp]))
        print(cnf_matrix_train)

class LCRF():

    def __init__(self):
        self.fbp = LearningSystemFBP()
        self.fbp.load()
        self.tp = LearningSystemTP()
        self.tp.load()
        self.list_of_cp = None
        self.pred_prob_fbp = None
        self.pred_prob_tp = None
        self.list_of_img = None
        self.min_n_img = 2
        self.seq_is_complete = False
        self.n=10

    def set_params(self,seq_is_complete = False):
        self.seq_is_complete = seq_is_complete

    def makeNewPrediction(self):
        list_of_cp = self.list_of_cp
        pred_prob = self.pred_prob_fbp[-1]
        pred_prob[pred_prob==0]=1e-50
        tp_prob = self.pred_prob_tp[-1]
        if list_of_cp is None:
            self.list_of_cp = [[[None, -np.log2(ii)] for ii in pred_prob]]
        else :
            temp_oc = []
            ll = list_of_cp[-1]
            for idx, ii in enumerate(pred_prob):
                tidx = idx
                ts = -np.log2(ii)+ll[idx][1]
                for idxl in range(idx+1):
                    tn = -np.log2(ii)+ll[idxl][1]
                    if idxl < idx:
                        tn = tn+1-tp_prob[1]
                    else:
                        tn = tn+tp_prob[1]
                    if ts > tn:
                        ts = tn
                        tidx = idxl
                temp_oc.extend([[tidx,ts]])
            list_of_cp.extend([temp_oc])
            self.list_of_cp = list_of_cp
            # return list_of_cp

    def add_new_img(self,img):
        if img.shape != (480,480):
            print("image must have size 480x480")
        else:
            if self.list_of_img is None:
                self.list_of_img = np.array([img])
                self.pred_prob_fbp = np.array([self.fbp.predict_proba(img=img)])
                self.pred_prob_tp = np.array([self.tp.pred_prob(img1=None,img2=img)])
            else :
                self.pred_prob_tp = np.append(self.pred_prob_tp,[self.tp.pred_prob(img1=self.list_of_img[-1,:,:],img2=img)],axis=0)
                self.list_of_img = np.append(self.list_of_img,[img],axis=0)
                self.pred_prob_fbp = np.append(self.pred_prob_fbp,[self.fbp.predict_proba(img=img)],axis=0)

            self.makeNewPrediction()

    def pred_img(self,n=1):
        if self.list_of_img is None:
            print("cannot make a prediction without input sequance of image")
        elif n==0 and self.list_of_img.shape[0] == 1:
            print("cannot make a prediction using one image, at least the sequance contain "+self.n.__str__()+" of contigous image")
        elif n== self.list_of_img.shape[0]-1:
            if self.seq_is_complete:
                list_of_cp = self.list_of_cp
                ll = list_of_cp[n]
                minV = ll[0][1]
                iidx = 0
                for idx, ii in enumerate(ll):
                    if ii[1]<minV:
                        iidx = idx
                        minV = ii[1]
                return iidx+1
            else:
                print("the sequance is not complete thus cannot make a prediction of last frame, "
                      "the predicition only be made for frame t-"+self.n.__str__())
        elif n < self.list_of_img.shape[0]-self.n:
            list_of_cp = self.list_of_cp
            nn = self.list_of_img.shape[0]-1
            ll = list_of_cp[nn]
            minV = ll[0][1]
            iidx = 0
            for idx, ii in enumerate(ll):
                if ii[1]<minV:
                    iidx = idx
                    minV = ii[1]
            while nn > n+1:
                iidx = self.list_of_cp[nn][iidx][0]
                nn-=1
            return list_of_cp[nn][iidx][0]+1
        else:
            if self.seq_is_complete:
                list_of_cp = self.list_of_cp
                nn = self.list_of_img.shape[0]-1
                ll = list_of_cp[nn]
                minV = ll[0][1]
                iidx = 0
                for idx, ii in enumerate(ll):
                    if ii[1]<minV:
                        iidx = idx
                        minV = ii[1]
                while nn > n+1:
                    iidx = self.list_of_cp[nn][iidx][0]
                    nn-=1
                return list_of_cp[nn][iidx][0]+1
            else:
                print("the sequance is not complete thus cannot make the prediction of this frame, "
                      "the predicition only be made for frame t-"+self.n.__str__())

class testLCRF():

    def __init__(self):
        self.machine = LCRF()

    def readAndValidateImg(self, fname):
        if fname.find("Frame") != -1 and fname.find(".png") != -1:
            img = imread(fname, as_grey=False)
            if img.shape.__len__() == 2:
                return img
            else:
                return None
        else:
            return None

    def runningTest(self):
        path = "test\\E03"
        fullPath = []
        for className in listdir(path):
            fullPath.extend([path+"\\"+className])
        n_in_img = 0
        n_pred = 0
        arr = []
        y = []
        for dir in fullPath:
            print(dir)
            yy = dir[-1]
            for imgFname in listdir(dir):
                # print(imgFname)
                img = self.readAndValidateImg(dir+"\\"+imgFname)
                if img is not None:
                    y.extend([yy])
                    self.machine.add_new_img(img)
                    n_in_img += 1

                    pp = self.machine.pred_img(n_pred)
                    if pp is not None:
                        arr.extend([pp])
                        # print(pp)
                        n_pred+=1
        self.machine.set_params(seq_is_complete=True)
        while n_pred != n_in_img:
            pp = self.machine.pred_img(n_pred)
            if pp is not None:
                arr.extend([pp])
                # print(pp)
                n_pred+=1
        arr = np.array(arr)
        print(arr.shape)
        cnf_matrix_train = confusion_matrix(y, [ii.__str__() for ii in arr])
        print(cnf_matrix_train)

if __name__ == "__main__":
    ls = LearningSystemFBP(loadDataSet=True)
    ls.build(retrain=True)
    print(ls.selectedFeature)
    #
    # ls = LearningSystemTP(loadDataSet=True)
    # ls.build(retrain=True)
    # ls.report()
    #
    ls = LearningSystemCRF()
    ls.testPredSeq()

    ls = testLCRF()
    ls.runningTest()
