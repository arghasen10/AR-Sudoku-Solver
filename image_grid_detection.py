import cv2

img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,200,255,cv2.RETR_EXTERNAL)[1]
thresh2 = cv2.threshold(gray,255,255,cv2.RETR_EXTERNAL)[1]
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cnts = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
i=0
for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area <50 :
        continue
    if area >300:
        continue

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
cv2.imshow('clear',thresh2)
cv2.imshow('sudoku',img)
cv2.waitKey(0)
cv2.destroyAllWindows()