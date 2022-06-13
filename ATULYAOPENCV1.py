import cv2
import numpy as np
import cv2.aruco as aruco
import imutils
import math






width= 1200
hieght = 750
def arucoDetect(img):
    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    key = getattr(aruco , f'DICT_5X5_250')
    dict = aruco.Dictionary_get(key)
    arucoparam = aruco.DetectorParameters_create()
    (corners,ids,rejected) = aruco.detectMarkers(img ,dict ,parameters = arucoparam)
    return corners , ids , rejected

def aruco_corners(img):
    (c,i,r) = arucoDetect(img)
    if len(c)>0:
        i = i.flatten()
        for(markercorner , markerid) in zip(c ,i):
            corner = markercorner.reshape((4,2))
            (tl,tr,br,bl)=corner
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
        return tl,tr,br,bl

def distance(x1 , y1 ,x2,y2):
    dist =int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return dist

#defining function to crop aruco markers , removing extra padding
def crop(img):
    tl,tr,br,bl=aruco_corners(img)
    arr = [tl,tr,br,bl]
    xmax = arr[0][0]
    xmin = arr[0][1]
    ymax = arr[1][0]
    ymin = arr[1][1]
    for i in arr:
        if i[0]>xmax:
            xmax =i[0]
        if i[0]<xmin:
            xmin =i[0]
        if i[1]>ymax:
            xmax =i[1]
        if i[1]<ymin:
            xmax =i[1]
    fin =img[ymin:ymax,xmin:xmax]
    return fin


#reading images of aruco markers
a1 = cv2.imread('LMAO.jpg')
a2 = cv2.imread('XD.jpg')
a3 = cv2.imread('Ha.jpg')
a4 = cv2.imread('HaHa.jpg')

# print(coordinates)
#rotaion of aruco mar
Ra1 = imutils.rotate_bound(a1,15.5)
Ra2 = imutils.rotate_bound(a2,-12.4)
Ra4 = imutils.rotate_bound(a4,14.9)

#removing extra padding in arucos
cra1 = crop(Ra1)
cra2 = crop(Ra2)
cra3 = crop(a3)
cra4 = crop(Ra4)

#finding dimensions of squares using distance formula
dx1 =distance(137 ,19 ,412 ,78)
dy1 =distance(412 ,78 ,346 , 322)
dx2 =distance(960 ,283 ,1161, 354)
dy2 =distance(1161 ,354 ,1071 ,522)
dx3 =distance(799 ,46,1042 , 45)
dy3 =distance(1042 ,45 ,1043 ,259)
dx4 =distance(401,689 ,720,534 ,)
dy4 =distance(646 ,252 ,720 ,534)

#resizing arucos
A1 = cv2.resize(cra1 ,(dx1 ,dy1))
A2 = cv2.resize(cra2 ,(dx2 ,dy2))
A3 = cv2.resize(cra3 ,(dx3 ,dy3))
A4 = cv2.resize(cra4 ,(dx4 ,dy4))

#getting final aruco images after tilting according to the squares
fa1 = imutils.rotate_bound(A1,12.1)
fa2 = A2
fa3 = imutils.rotate_bound(A3,15.1)
fa4 = imutils.rotate_bound(A4,22.9)



#shape and colour detction
img = cv2.imread('CVtask.jpg')
img1 = cv2.resize(img, (1200 , 750))
grayim = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayim, 240, 255, cv2.THRESH_BINARY)
contours, xyz = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_COMPLEX
src_mask = np.zeros((width,hieght) ,img1.dtype)



for contour in contours:
    approx = cv2.approxPolyDP(contour , .01*cv2.arcLength(contour ,True) ,True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]


    if len(approx)== 4:
        x, y ,w ,d = cv2.boundingRect(approx)
        aspectRatio = float(w)/float(d)


        if (aspectRatio >=0.99)and(aspectRatio<=1.20):
            cv2.drawContours(img1, [contour], 0 , (0, 255, 0), 3)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)



            ###checking the color at the center of the countour with the colorall over into the squares
            if any(img[int(rect[0][1]), int(rect[0][0])]) == (81, 209, 144):
                output = cv2.seamlessClone(fa1, img1, src_mask,(rect[0][0] , rect[0][1]), cv2.NORMAL_CLONE)
               # print(type(output)
               #  print(len(contour))
            if any(img[int(rect[0][1]), int(rect[0][0])]) == (9, 127, 240):
                output = cv2.seamlessClone(fa2, img1, src_mask, rect[0], cv2.NORMAL_CLONE)
               ## print(type(output))
                # print(len(contour))

            if any(img[int(rect[0][1]), int(rect[0][0])]) == (0, 0, 0):
                output = cv2.seamlessClone(fa3, img1, src_mask, rect[0], cv2.NORMAL_CLONE)
                #print(type(output))
                # print(len(contour))

            if any(img[int(rect[0][1]), int(rect[0][0])]) == (210, 222, 228):
                output = cv2.seamlessClone(fa4, img1, src_mask, rect[0], cv2.NORMAL_CLONE)
                #print(type(output))
                # print(len(contour))


            n = approx.ravel()
            i = 0

            #to get the coordinates of squares
            for j in n:
                if (i <= 4):
                    x = n[i]
                    y = n[i + 1]
                    # String containing the co-ordinates.
                    string = str(x) + ", " + str(y)
                    cv2.putText(img1, string, (x, y), font, 0.3, (255, 45, 255), 1)
                    i = i + 1

# print(type(output))
cv2.imshow( 'final'  , output )

if cv2.waitKey(0) & 0xFF == ord('q'):
   cv2.destroyAllWindows()

