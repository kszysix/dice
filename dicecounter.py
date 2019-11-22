import cv2
from shapedetector import ShapeDetector
import math
import numpy as np
import os

filename = [ "d1.jpg","d2.jpg","d3.jpg","d4.jpg","d5.jpg",
            "d6.jpg","d7.jpg","d8.jpg","d9.jpg","d10.jpg",
            "d11.jpg","d12.jpg","d13.jpg","d14.jpg","d15.jpg",
            "d16.jpg","d17.jpg","d18.jpg","d19.jpg","d20.jpg",
            "d21.jpg","d22.jpg","d23.jpg","d24.jpg","d25.jpg",
            "d26.jpg","d27.jpg","d28.jpg","d29.jpg","d30.jpg",
            "d31.jpg"
            ]

filename1 = ["d17.jpg"]

path = "input"

photos = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' or '.png' in file:
            photos.append(file)
print("Images found: ")
for f in photos:
    print(f)


#--------------PRZYGOTOWANIE-PLIKU------------------

for fi,file in enumerate(photos):

    #file = path + file
    image = cv2.imread(os.path.join(path,file))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    kropki = gray
    #kropki = cv2.threshold(kropki, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
     cv2.THRESH_BINARY,781, 8)

    kropki = cv2.adaptiveThreshold(kropki, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
     cv2.THRESH_BINARY,341, 7)

    #thresh = cv2.erode(thresh,(15,15),iterations=5)
    thresh = cv2.erode(thresh,(15,15),iterations=20)
    #cv2.imwrite("thr"+str(fi)+".jpg",thresh)
    #continue
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
    shapek = []
    shapec = []
    r = 0
    p = 0 

#---------------KOSTKI-------------------

    for i,c in enumerate(cnts):
        #odrzucanie zbyt małych i zbyt dużych konturów
        if((cv2.arcLength(c,True)<500 or cv2.arcLength(c,True)>4000) ):
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
                shapek.append("Dice "+str(r))
                r = r+1
                
  

            if(shape == "niet"):
                continue


#-----------KROPKI------------------
    r = 0
    p = 0 
    for i,c in enumerate(krops):
        #odrzucanie zbyt małych i zbyt dużych konturów
        if((cv2.arcLength(c,True)<50 or cv2.arcLength(c,True)>4000) ):
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
                shapec.append("Dot "+str(p))
                p = p+1

            if(shape == "niet"):
                continue
#------------LICZENIE----------------------
    wynik = []

    for i,kost in enumerate(kostki):
     
               #print("tak")

        owk = [] #oczka w kostce
        c = kost
        # mincx = c[0][0][0]
        # maxcx = c[0][0][0]
        # mincy = c[0][0][1]
        # maxcy = c[0][0][1]

        # for ic,k in enumerate(c):
        #     if(c[ic][0][0]<mincx):
        #         mincx = c[ic][0][0]

        #     if(c[ic][0][0]>maxcx):
        #         maxcx = c[ic][0][0]
            
        #     if(c[ic][0][1]<mincy):
        #         mincy = c[ic][0][1]
            
        #     if(c[ic][0][1]>maxcy):
        #         maxcy = c[ic][0][1]
        #print(mincx,maxcx,mincy,maxcy)

        for k, ocz in enumerate(oczka):
            if(cv2.arcLength(c,True)/14>cv2.arcLength(ocz,True) or cv2.arcLength(c,True)/4<cv2.arcLength(ocz,True) ):
                continue

            M = cv2.moments(ocz)
            if M["m00"] != 0:  #w celu nie dzielenia przez 0
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                #print(cX,cY)
                #if(cX>mincx and cX<maxcx and cY>mincy and cY<maxcy):
                dist = cv2.pointPolygonTest(kost,(cX,cY),False)
                if(dist != -1):
                    owk.append(k)
                    
                    co = ocz.astype("float")
                    co = ocz.astype("int")
                    cv2.drawContours(image,[co],-1,(80,240,20),4)
                    cv2.putText(image,shapec[k],(cX,cY),cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (140,240,20),2)
        if len(owk)>0:
            M = cv2.moments(kost)
            if M["m00"] != 0:  #w celu nie dzielenia przez 0
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))

                ck = kost.astype("float")
                ck = kost.astype("int")
                cv2.drawContours(image, [ck], -1, (0, 0, 255), 4)
                cv2.putText(image, shapek[i], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1, (40, 50, 250), 2)
        wynik.append(owk)
        #print("OK")

#-------------------ZAPIS---------------------
    it = 1
    for i,w in enumerate(wynik):
        if (len(w)!=0):
            opis = "Dice number: "+str(i)+", dots: "+str(len(w))+", dots numbers: "+str(w)
            org = (50,50+it*80)
            it=it+1
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 2
            color = (50, 180, 180) 
            thickness = 2
            cv2.putText(image,opis,org,font,fontScale,color,thickness)
    name = "./output/dice" + str(fi)+".jpg"
    #name = "aapoly.jpg" + str(fi)+".jpg"
    cv2.imwrite(name,image)
    #print("OK")

    for i,w in enumerate(wynik) :
        if (len(w)!=0):
            print("Kostka nr ",i,", ilość oczek: ",len(w), " zawartosc ",w)
