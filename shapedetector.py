import cv2
import math

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        #z założenia odrzucamy kontur
        shape = "niet"

        #pobieramy długość konturu w pikselach, czyli sumy odległości między punktami z findContours()
        clen = cv2.arcLength(c, True)

        #epsilion = 0.04*clen oznacza możliwą odchyłkę od oryginału
        #zazwyczaj epsilon = 0.01*clen ale przy zdjęciach są duże odchyłki w wyszukiwanych konturach
        #używany jest algorytm Ramer–Douglas–Peucker do przybliżenia
        #True oznacza, że kontur będzie zamknięty
        #approx jest tablicą punktów przybliżonego konturu
        approx = cv2.approxPolyDP(c, 0.04 * clen, True)
        
        #dla kwadratu przy odpowiednim epsilonie powinny być 4 punkty
        if len(approx) == 4:

            #zapisujemy współrzędne wierzchołków prostokąta na obliczanym czworokącie
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            #oceniamy jak blisko jest temu czworokątowi do kwadrata
            if (ar >= 0.8 and ar <= 1.2):

                #------wypukłość-------
                #pierwszy test wypukłości
                c1 = c[0][0]
                c2 = c[int(0.25*len(c))][0]
                c3 = c[int(0.5*len(c))][0]
                c4 = c[int(0.75*len(c))][0]
                #print(c1,c2,c3,c4)
                dl13 = int(math.sqrt((c1[0]-c3[0])**2+(c1[1]-c3[1])**2))
                dl24 = int(math.sqrt((c2[0]-c4[0])**2+(c2[1]-c4[1])**2))
                
                bok = int(cv2.arcLength(c,True)/4)
                blad = 4*int(bok*0.1)

                #print(dl13,dl24,bok,blad)
                #sprawdzamy czy z pewnym błędem, odległości między punktów są długości boków
                if((dl13>bok-blad) and (dl13<bok+blad) and (dl24>bok-blad) and (dl24<bok+blad)):
                    shape = "Square"
                    #print("S:" ,dl13,dl24,bok,cv2.arcLength(c,True))


        else:
            approx = cv2.approxPolyDP(c, 0.04 * clen, True)
            if len(approx)>4:
                
                #---sprawdzanie kulistości, czyli odległości konturu od środka
                shape = "circle"
                M = cv2.moments(c)
                if M["m00"] != 0:  #w celu nie dzielenia przez 0
                    cX = int((M["m10"] / M["m00"]))
                    cY = int((M["m01"] / M["m00"]))

                    radius = cv2.arcLength(c,True)/6.28
                    #print("radius ",radius)

                    for i,cs in enumerate(c):
                        dlrc = int(math.sqrt((cs[0][0]-cX)**2+(cs[0][1]-cY)**2))
                        #dlrc - odległość od środka do punktu konturu
                        if(dlrc > radius*1.4 or dlrc < radius*0.6):
                            shape = "niet"


                    #if(shape == "circle"):
                        #print("C:" ,dlrc,radius,cv2.arcLength(c,True))   

        return [shape, c]

