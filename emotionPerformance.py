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
import sys
import time
import datetime
from subprocess import check_output
from sklearn.metrics import confusion_matrix

t1 = datetime.datetime.now()
fileName = sys.argv[1]
faceDet = cv2.CascadeClassifier("/Users/yaminiaggarwal/Documents/temp/haarcascade_frontalface_default.xml")

trained_file="/Users/yaminiaggarwal/Documents/temp/trainedmodel.pkl"
y_actual="/Users/yaminiaggarwal/Documents/temp/y1.csv"
y_pred="/Users/yaminiaggarwal/Documents/temp/y2.csv"

#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
emotions = ["neutral", "anger", "disgust", "happy", "surprise"] #Define emotion order
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/yaminiaggarwal/Documents/temp/shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []
t2 = datetime.datetime.now()
print("load time", t2-t1)

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
        #print("shape", shape)
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
        print("landmarks_vectorised", len(landmarks_vectorised))
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

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def f_score(label, confusion_matrix):
    precisioni = precision(label, confusion_matrix)
    recalli = recall(label, confusion_matrix)
    return (2 * precisioni * recalli) / (precisioni + recalli) 

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows
def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def accuracy(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_accuracy = 0
    for i in range(rows):
        sum_of_accuracy += confusion_matrix[i][i]
    return sum_of_accuracy / confusion_matrix.sum()    

def update_model():
    accur_lin = []
    clf = SVC(kernel='linear', probability=True, tol=1e-2)#, verbose = True) #Set the classifier as a support vector machines with linear kernel
    finalClf = clf;

    print (clf)
    maxAccuracy = -1
    for i in range(0,1):  # it is upto 10
        #print("Making sets %s" %i) #Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = make_sets()

        npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        t1 = datetime.datetime.now()
        clf.fit(npar_train, training_labels)
        t2 = datetime.datetime.now()
        print("time for learning: ", t2-t1)
        train_out = clf.predict(npar_train)
        train_lin = clf.score(npar_train, training_labels)
        npar_pred = np.array(prediction_data)
        npar_predlabs = np.array(prediction_labels)
        

        pred_out = clf.predict(npar_pred)
        
        print("y actual: ", len(npar_predlabs))
        print(npar_predlabs)

        print("y predicted: ", len(pred_out))
        print(pred_out)

         
        print("Confusion matrix")
        cm = confusion_matrix(npar_predlabs, pred_out)

        print(np.matrix(cm))
        for i in range(0,5):
            print("precision for ", emotions[i], precision(i,cm))
            print("recall for ", emotions[i], recall(i,cm))
            print("F-score for ", emotions[i], f_score(i,cm))

        print("Average precision" , precision_macro_average(cm))    
        print("Average recall" , recall_macro_average(cm))
        print("Average accuracy" , accuracy(cm))

        #print(confusion_matrix)
        '''for i in range(len(pred_out)):
            if pred_out[i] != npar_predlabs[i]:
                print ("image no.: ", i, "emotion predicted :", emotions[pred_out[i]], "actual emotion :", emotions[npar_predlabs[i]])
                print (i, emotions[pred_out[i]], emotions[npar_predlabs[i]])
        '''
        pred_lin = clf.score(npar_pred, prediction_labels)
        print ("linear: ", pred_lin)
        accur_lin.append(pred_lin) #Store accuracy in a list
        if pred_lin > maxAccuracy:
            maxAccuracy = pred_lin;
            finalClf = clf;

    #print("Dumping model into the trained file with accuracy: ", maxAccuracy)
    with open(trained_file, 'wb') as file:
        pickle.dump(finalClf, file)


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
    if (video_capture.isOpened() == False): 
        print("Error opening video stream or file")
    
    # Calculate frames per second
    t1 = datetime.datetime.now()
    num_frames = 0
    #For Windows
    #a = str(check_output('ffprobe -i  "'+fileName+'" 2>&1 |findstr "Duration"',shell=True)) 

    #For Linux
    a = str(check_output('ffprobe -i  "'+fileName+'" 2>&1 |grep "Duration"',shell=True)) 
    a = a.split(",")[0].split("Duration:")[1].strip()
    h, m, s = a.split(':')
    duration = int(h) * 3600 + int(m) * 60 + float(s)
    #print("Duration of video: ", duration)
    
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))    
    #print("Total frames: ", num_frames) 
    frames_in_sec = 25
    frames_to_process = 5
    try:
        frames_in_sec = num_frames/duration
    except:
        print("Frames in 1 sec is set to default(25)")

    print("Frames in 1 sec: ", frames_in_sec) 
    t2 = datetime.datetime.now()
    print("Time for calculating fps: ", t2-t1)
 

    try:
        # Load from file
        with open(trained_file, 'rb') as file:  
            clf = pickle.load(file)
            #print("Classifier loaded ", clf)
    except:
        print("no xml found. Using --update will create one.")
        update_model()
    #t2 = datetime.datetime.now()
    #print("load classifier time: ", t2-t1)
    
    count = 0
    countFace = 0
    countEmotion = [0,0,0,0,0]
    framesProcessed = 0
    nth_frame = int(frames_in_sec/frames_to_process)
    
    while (video_capture.isOpened()):
        #t1 = datetime.datetime.now()
        ret, frame = video_capture.read()
        #t2 = datetime.datetime.now()
        #print("read frame time: ", t2-t1)
        if ret == False:
            break
        #t1 = datetime.datetime.now()    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #t2 = datetime.datetime.now()
        #print("grayscale time: ", t2-t1)
        count = count + 1
        
        #print("nth frame: ", nth_frame)
        #t1 = datetime.datetime.now()
        #Detect face using 4 different classifiers
        if (count ==1 or (count%nth_frame)==0):
            framesProcessed = framesProcessed+1
            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
            #t2 = datetime.datetime.now()
            #print("detect face time: ", t2-t1)
            #Go over detected faces, stop at first detected face, return empty if no face.
            #print("face: ", face)
            if len(face) == 1:
                facefeatures = face
            else:
                facefeatures = ""
            emotion = "unknown"
            #Cut and save face
            #t1 = datetime.datetime.now()
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
                    '''cv2.putText(img = frame, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))
                    font=70
                    for index in range(len(predictProbaData)):
                        label = "%s : %.5f" %(emotions[index], predictProbaData[index])
                        #print ("label :", label)
                        if emotionIndex == index:
                            cv2.putText(img = frame, text = label, org = (20,font), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 255))
                        else:
                            cv2.putText(img = frame, text = label, org = (20,font), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0))
                        font=font+20
                    '''   
                except:
                   pass #If error, pass file
                   #print("exception occurred in resizing")
                   cv2.putText(img = frame, text = emotion, org = (30,40), fontFace = cv2.FONT_ITALIC, fontScale = 1, color = (0, 0, 255))    

            #cv2.imshow("image", frame) #Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
                    break
            #t2 = datetime.datetime.now()
            #print("predict time: ", t2-t1)

    #t1 = datetime.datetime.now()    
    # When everything is done, release the capture
    print("Frames Processed: ", framesProcessed)
    video_capture.release()
    cv2.destroyAllWindows()
    #t2 = datetime.datetime.now()
    #print("release capture time: ", t2-t1)
    print("No. of frames: ", count)
    return countEmotion

t1 = datetime.datetime.now()
countEmotion = capture_image();
#update_model();
t2 = datetime.datetime.now()
print("Total time: ", t2-t1)
print (countEmotion)

    
