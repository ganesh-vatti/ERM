import cv2
import glob
import random
import math
from imutils.video import FPS
from imutils.video import FileVideoStream
import imutils
import numpy as np
import dlib
import pickle
from sklearn.svm import SVC
from sklearn import preprocessing
import sys
import time
import datetime

fileName = sys.argv[1]
faceDet = cv2.CascadeClassifier("/Users/yaminiaggarwal/Documents/temp/haarcascade_frontalface_default.xml")

trained_file="/Users/yaminiaggarwal/Documents/temp/trainedmodel.pkl"


#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
emotions = ["neutral", "anger", "disgust", "happy", "surprise"] #Define emotion order
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/yaminiaggarwal/Documents/temp/shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-2)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

finalClf = clf;

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob('*dataset/%s/*' %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    #print ("get_files" , len(training) , len(prediction))
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1: 
        data['landmarks_vectorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        #print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
                # print("landmarks_vectorised", data['landmarks_vectorised'])
    return training_data, training_labels, prediction_data, prediction_labels   

def update_model():
    accur_lin = []
    print (clf)
    maxAccuracy = -1
    for i in range(0,5):  # it is upto 10
        #print("Making sets %s" %i) #Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = make_sets()

        npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        #print("train_X %s" %npar_train)
        #print(npar_train.shape)
        #print("train_Y %s" %npar_trainlabs)

        #train_scaled = preprocessing.scale(npar_train)
        #print("training SVM linear %s" %i) #train SVM
        clf.fit(npar_train, training_labels)

        train_out = clf.predict(npar_train)
        train_lin = clf.score(npar_train, training_labels)
        #print ("accuracy on training: ", train_lin)

        #print("getting accuracies %s" %i) #Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        npar_predlabs = np.array(prediction_labels)
        #prediction_scaled = preprocessing.scale(npar_pred)

        pred_out = clf.predict(npar_pred)

        for i in range(len(pred_out)):
            if pred_out[i] != npar_predlabs[i]:
                print ("image no.: ", i, "emotion predicted :", emotions[pred_out[i]], "actual emotion :", emotions[npar_predlabs[i]])
                print (i, emotions[pred_out[i]], emotions[npar_predlabs[i]])

        pred_lin = clf.score(npar_pred, prediction_labels)
        #print ("linear: ", pred_lin)
        accur_lin.append(pred_lin) #Store accuracy in a list
        #emone  = [npar_pred[2]]
        #emLabelOne = clf.predict(emone)
        #print ("emLabelOne: ", emLabelOne)
        if pred_lin > maxAccuracy:
            maxAccuracy = pred_lin;
            finalClf = clf;

    #print("Dumping model into the trained file with accuracy: ", maxAccuracy)
    with open(trained_file, 'wb') as file:
        pickle.dump(finalClf, file)
    #print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs


# start capturing the video

def start_video():
    video_capture = cv2.VideoCapture(0) #Webcam object
    frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        get_landmarks(clahe_image)

        prediction_data = []
        if data['landmarks_vectorised'] == "error":
           print("no face detected on this one")
        else:
           prediction_data.append(data['landmarks_vectorised'])
           npar_pred = np.array(prediction_data) #Turn the training set into a numpy array for the classifier
           emotionIndex =  clf.predict(npar_pred)
           print ("emotionIndex :", emotionIndex)
           emotion = emotions[emotionIndex[0]]
           print ("emotion :", emotion)
           cv2.putText(img = frame, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))    
       
        cv2.imshow("image", frame) #Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break
        
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def predict_image(item):
    image = cv2.imread(item)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    get_landmarks(clahe_image)
    prediction_data = []
    if data['landmarks_vectorised'] == "error":
       print("no face detected on this one")
    else:
       prediction_data.append(data['landmarks_vectorised'])
       npar_pred = np.array(prediction_data) #Turn the training set into a numpy array for the classifier
       emotionIndex =  clf.predict(npar_pred)
       print ("emotionIndex :", emotionIndex)
       emotion = emotions[emotionIndex[0]]
       print ("emotion :", emotion)
       cv2.putText(img = image, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))    
    
    cv2.imshow(item, image) #Display the frame
    #if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
    # break

def predict2_image(gray, clf2):
##    image = cv2.imread(item)
##    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    emotion = "unknown"
    clahe_image = clahe.apply(gray)
    #print("getting landmarks")
    get_landmarks(clahe_image)
    #print("got landmarks")
    prediction_data = []
    if data['landmarks_vectorised'] == "error":
       print("no face detected on this one")
    else:
       #print("Appending to prediction data") 
       prediction_data.append(data['landmarks_vectorised'])
       #print("creating nparray of prediction data ") 
       npar_pred = np.array(prediction_data) #Turn the training set into a numpy array for the classifier
       #print("classifier for prediction", clf)
       try:
           emotionIndex =  clf2.predict(npar_pred)
       except:
           pass #If error, pass file
           #print("exception occurred in prediction")    
       #print ("emotionIndex :", emotionIndex)
       emotion = emotions[emotionIndex[0]]
       predict_proba = clf2.predict_proba(npar_pred)
       #print("predict_proba :", predict_proba)
       #print ("emotion :", emotion)
    return emotionIndex[0], predict_proba[0]


def capture_image():
    #video_capture = cv2.VideoCapture(0) #Webcam object
    #video_capture = cv2.VideoCapture('https://youtu.be/lB5KBM17pIs') #Webcam object
    video_capture = cv2.VideoCapture(fileName) #Webcam object
    #fvs = FileVideoStream(fileName).start()
    #time.sleep(1.0)
    #fps = FPS().start()
    if (video_capture.isOpened() == False): 
        print("Error opening video stream or file")

    try:
        # Load from file
        with open(trained_file, 'rb') as file:  
            clf = pickle.load(file)
            #print("Classifier loaded ", clf)
    except:
        print("no xml found. Using --update will create one.")
        update_model()

    count = 0
    countFace = 0
    countEmotion = [0,0,0,0,0]

    while (video_capture.isOpened()):
    #while fvs.more():
        ret, frame = video_capture.read()
        #time.sleep(0.001)
        #frame = fvs.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count = count + 1
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
##        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
##        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
##        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
            #print("Face detected ", facefeatures)
##        elif len(face_two) == 1:
##            facefeatures = face_two
##        elif len(face_three) == 1:
##            facefeatures = face_three
##        elif len(face_four) == 1:
##            facefeatures = face_four
        else:
            facefeatures = ""
            #print("Face Features Empty ", facefeatures)
            
        emotion = "unknown"
        #Cut and save face

        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            countFace = countFace+1
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                emotionIndex,  predictProbaData = predict2_image(out, clf)
                #print("EmotionIndex :" , emotionIndex)
                emotion = emotions[emotionIndex]
                countEmotion[emotionIndex] = countEmotion[emotionIndex] + 1

                #print("Emotion :" , emotion)
                #print("predictProbaData :" , predictProbaData)
                #print("predictProbaData[0] :" , predictProbaData[0])
                #print("predictProbaData[1] :" , predictProbaData[1])

                
                #cv2.imwrite("dataset\\test\\%s.jpg" %(count), out) #Write image
                cv2.putText(img = frame, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))
                font=70
                for index in range(len(predictProbaData)):
                    label = "%s : %.5f" %(emotions[index], predictProbaData[index])
                    #print ("label :", label)
                    if emotionIndex == index:
                        cv2.putText(img = frame, text = label, org = (20,font), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 255))
                    else:
                        cv2.putText(img = frame, text = label, org = (20,font), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0))
                    font=font+20
            except:
               pass #If error, pass file
               #print("exception occurred in resizing")
               cv2.putText(img = frame, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))    

        cv2.imshow("image", frame) #Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
                break
        #fps.update()
    #print ("Count of faces: ", countFace)
    #print ("Count of each emotion: ", countEmotion)
    #strCountEmotion = " ".join(str(x) for x in countEmotion)

    #fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    #fvs.stop()
    return countEmotion

'''
def my_range(start, end, step):
    while start <= end:
        yield start
        start *= step

#cList = my_range(1e-2,1e2,10)    
#tolList = my_range(1e-4,1e-1,10)

cList = (0.01,0.1,1,10)   
tolList = (0.0001,0.001,0.01,0.1)

for x in cList:
    for y in tolList:
        clf = SVC(C=x, kernel='linear', probability=True, tol=y)
        update_model()  
'''          

#update_model();

t1 = datetime.datetime.now()
countEmotion = capture_image();
t2 = datetime.datetime.now()
print("Total time: ", t2-t1)
print (countEmotion)
##predict_image('dataset/anger/2.jpg')
##predict_image('dataset/happy/2.jpg')
##predict_image('dataset/neutral/2.jpg')
##predict_image('dataset/surprise/2.jpg')
##predict_image('dataset/disgust/2.jpg')
    
