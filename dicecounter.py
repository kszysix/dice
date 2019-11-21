import cv2
from shapedetector import ShapeDetector
import math
import numpy as np

filename = [ "d1.jpg","d2.jpg","d3.jpg","d4.jpg","d5.jpg",
            "d6.jpg","d7.jpg","d8.jpg","d9.jpg","d10.jpg",
            "d11.jpg","d12.jpg","d13.jpg","d14.jpg","d15.jpg",
            "d16.jpg","d17.jpg","d18.jpg","d19.jpg","d20.jpg",
            "d21.jpg","d22.jpg","d23.jpg","d24.jpg","d25.jpg",
            "d26.jpg","d27.jpg","d28.jpg","d29.jpg","d30.jpg",
            "d31.jpg"
            ]

filename1 = ["d1.jpg"]

for fi,file in enumerate(filename):

    image = cv2.imread(file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    kropki = gray
    #kropki = cv2.threshold(kropki, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
     cv2.THRESH_BINARY,781, 8)

    kropki = cv2.adaptiveThreshold(kropki, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
     cv2.THRESH_BINARY,341, 7)

    #thresh = cv2.erode(thresh,(15,15),iterations=5)
    thresh = cv2.erode(thresh,(15,15),iterations=20)
    #cv2.imwrite('thr.jpg',thresh)
    #tworzenie kernela do 'opening'
    ksize = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    #opening

    kropki = cv2.morphologyEx(kropki, cv2.MORPH_OPEN, kernel) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #znajdowanie konturów
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    krops = cv2.findContours(kropki.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #pobieranie puntków konturów
    cnts = cnts[1]
    krops = krops[1]
    #tree = cnts

    #obiekt klasy ShapeDetector
    sd = ShapeDetector()
    kostki = []
    oczka = []
    r = 0
    p = 0 
    #----------KOSTKI-------------------
    for i,c in enumerate(cnts):
        #odrzucanie zbyt małych i zbyt dużych konturów
        if((cv2.arcLength(c,True)<100 or cv2.arcLength(c,True)>4000) ):
            continue
        
        #wyszukiwanie środka konturu
        M = cv2.moments(c)
        if M["m00"] != 0:  #w celu nie dzielenia przez 0
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))

            #użycie metody przypisującej kształt konturowi
            zwrot = sd.detect(c)
            shape = zwrot[0]
            if(shape == "Square"):
                #print("kostka")
                kostki.append(zwrot[1])
                shape = "Square"+str(r)
                r = r+1
                
                c = c.astype("float")
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 0, 255), 4)
                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

            if(shape == "niet"):
                continue


#-----------KROPKI------------------
    r = 0
    p = 0 
    for i,c in enumerate(krops):
        #odrzucanie zbyt małych i zbyt dużych konturów
        if((cv2.arcLength(c,True)<100 or cv2.arcLength(c,True)>4000) ):
            continue
        
        #wyszukiwanie środka konturu
        M = cv2.moments(c)
        if M["m00"] != 0:  #w celu nie dzielenia przez 0
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))

            #użycie metody przypisującej kształt konturowi
            zwrot = sd.detect(c)
            shape = zwrot[0]
            if(shape == "circle"):
                #print("oczka")
                oczka.append(zwrot[1])
                shape = "Circle"+str(p)
                p = p+1

                c = c.astype("float")
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 0, 255), 4)
                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2)

            if(shape == "niet"):
                continue

    #print("OK")
    #----ZAPIS------
    name = "./apolygs/polyg" + str(fi)+".jpg"
    #name = "poly.jpg" + str(fi)+".jpg"
    cv2.imwrite(name,image)
    #print("OK")

    wynik = []

    for i,kost in enumerate(kostki):
        
        owk = [] #oczka w kostce
        c = kost
        mincx = c[0][0][0]
        maxcx = c[0][0][0]
        mincy = c[0][0][1]
        maxcy = c[0][0][1]

        for i,k in enumerate(c):
            if(c[i][0][0]<mincx):
                mincx = c[i][0][0]

            if(c[i][0][0]>maxcx):
                maxcx = c[i][0][0]
            
            if(c[i][0][1]<mincy):
                mincy = c[i][0][1]
            
            if(c[i][0][1]>maxcy):
                maxcy = c[i][0][1]
        #print(mincx,maxcx,mincy,maxcy)

        for k, ocz in enumerate(oczka):
            if(cv2.arcLength(c,True)/10>cv2.arcLength(ocz,True) ):
                continue

            M = cv2.moments(ocz)
            if M["m00"] != 0:  #w celu nie dzielenia przez 0
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                #print(cX,cY)
                if(cX>mincx and cX<maxcx and cY>mincy and cY<maxcy):
                    owk.append(k)
                    #print("tak")
        #print(owk)
        wynik.append(owk)


    for i,w in enumerate(wynik) :
        if (len(w)!=0):
            print("Kostka nr ",i,", ilość oczek: ",len(w), " zawartosc ",w)
