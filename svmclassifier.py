import scipy
import csv as csv
import os
from PIL import Image
from PIL import ImageFilter
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
import re

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
#48.26 with smoothing test acc
#44.6 without smoothing test
#47.5 with smoothing test gauss
#44.18 without smoothing
classes = ['dogs', 'cats']
num_classes = len(classes)
#get full dataset
data_dir = 'noisytrain/'
CAT_TRAIN_DIR = data_dir+'cats/'
cat_train_images = [CAT_TRAIN_DIR+i for i in os.listdir(CAT_TRAIN_DIR)] # use this for training images

DOG_TRAIN_DIR = data_dir+'dogs/'
dog_train_images = [DOG_TRAIN_DIR+i for i in os.listdir(DOG_TRAIN_DIR)]

train_images = cat_train_images+dog_train_images

test_data_dir = 'test/'
CAT_TEST_DIR = test_data_dir+'cats/'
cat_test_images = [CAT_TEST_DIR+i for i in os.listdir(CAT_TEST_DIR)] # use this for test images

DOG_TEST_DIR = test_data_dir+'dogs/'
dog_test_images = [DOG_TEST_DIR+i for i in os.listdir(DOG_TEST_DIR)]
test_images = cat_test_images+dog_test_images
# test_images = train_images


#get labels
predictedlabels=[]
labels=[]
train_dogs=[]
train_cats=[]
images=[]
training_dict = {}
test_dict = {}


def train():            #gets correct class for each image

    for i in os.listdir(DOG_TRAIN_DIR):
        train_dogs.append(i)
        labels.append(1)
        training_dict[DOG_TRAIN_DIR+i] = 1

    for i in os.listdir(CAT_TRAIN_DIR):
        train_cats.append(i)
        labels.append(0)
        training_dict[CAT_TRAIN_DIR+i] = 0
    return labels


def test():            #gets correct class for each image

    for i in os.listdir(DOG_TEST_DIR):
        test_dict[DOG_TEST_DIR+i] = 1

    for i in os.listdir(CAT_TEST_DIR):
        test_dict[CAT_TEST_DIR+i] = 0

def getResults(predictedlabels, training_dict, dataset='Training'):    #outputs accuracy

    total=0
    newpredict={}
    for key, value in predictedlabels.items():

        if float(value)>0.5:
            newpredict[key] = 1
        else:
            newpredict[key] = 0


        newPredictedClass = newpredict[key]
        training_dictClass = training_dict[key]
        if newPredictedClass == training_dictClass:
            total+=1

    print(dataset,"Accuracy:",total,"/",len(training_dict),"* 100 =","{0:.3f}".format(total/len(training_dict)*100),"%")



def svm(y):             #trains svm on the training set

    results=[]
    pix_val=0
    new_images=[]
    placeholder=[]
    # svm = LinearSVC()
    svm = LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    # svm = SVC(kernel='linear')
    # svm = SVC(kernel='rbf')
    # svm = SVC(kernel='poly', degree=3)
    clf = CalibratedClassifierCV(svm, method='sigmoid')
    for i in train_images:
        pil_im = Image.open(i).convert('L')
        #pil_im = Image.open(i)
        size=64,64

        pil_im = pil_im.resize(size, Image.ANTIALIAS)
        #pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
        #adding gaussian blur to smooth out the noise
        #pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
        pix_val=pil_im.histogram()

        results.append(pix_val)


    #x should be an array (n_samples, n_features)
    clf = clf.fit(results,y)
    return clf


def getAccuracy(clf, images):          #test on the training set
    results={}
    total=0
    myfile = open('results.csv', 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')
    wr.writerow(["id","label"])
    for i in range(0,len(images)):
            j = images[i]
            pil_im = Image.open(j).convert('L')

            size=64,64

            pil_im = pil_im.resize(size, Image.ANTIALIAS)
            #pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
            # add gaussian blur to smooth out the noise
            #pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
            pix_val=pil_im.histogram()

            #results.append(clf.predict([pix_val]))
            x = str(clf.predict_proba([pix_val]))
            x=x[2:-2]

            x = re.split('\s+',x)
           # print(x)
            results[j] = x[2]

            if float(x[2]) >= 0.5:
                wr.writerow([j,'  Dog'])
            else:
                wr.writerow([j,'  Cat'])


    return results


y=train()   #get correct labels

clf=svm(y)   #get trained svm

imageinfo = getAccuracy(clf, train_images)       #get predictions for testing
getResults(imageinfo,training_dict, 'Training')

test()
imageinfo_test = getAccuracy(clf, test_images)       #get predictions for testing
getResults(imageinfo_test,test_dict, 'Test')