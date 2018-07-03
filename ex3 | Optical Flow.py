import cv2
import numpy as np
import matplotlib.pyplot as plt
drawing = False
mode = True
ix, iy = -1, -1
f = 6
def draw_circle (event, x, y, flags, param):
	global ix, iy, drawing, mode
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix, iy = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == True:
				cv2.rectangle( img, (ix, iy), (x, y), (0, 255, 255), f )
			else:
				cv2.circle( img, (x, y), 5, (0, 255, 255), f )
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle( img, (ix, iy), (x, y), (0, 255, 255), f )
		else:
			cv2.circle( img, (x, y), 5, (0, 255, 255), f )


img = np.zeros( (512, 512, 3), np.uint8 )

# cap = cv2.VideoCapture( 0 )
cap = cv2.VideoCapture( 0 )
print("Select an object to track")
print( 'click "q" to stop' )




ret, old_frame = cap.read( )
old_gray = cv2.cvtColor( old_frame, cv2.COLOR_BGR2GRAY )

mask = np.zeros_like( old_gray)


def de (event, x, y, flags, param):
	global ix, iy, drawing, mode
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix, iy = x, y
	return ix, iy




V=1
while (1):

	ret, frame = cap.read( )
	frame_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
	# img = cv2.add( frame, mask )
	img =frame_gray
	# cv2.namedWindow( 'image' )
	# cv2.setMouseCallback( 'image', draw_circle )
	cv2.namedWindow( 'image' )
	cv2.setMouseCallback( 'image', draw_circle )
	if(drawing):
		AAA =img

	if cv2.waitKey( 1 ) & 0xff == ord( 'q' ):
		BBB  = img
		break

	cv2.imshow( 'image', img )
cv2.destroyAllWindows( )
cap.release( )


plt.imshow(  AAA,"gray" )
plt.show()
# cv2.waitKey()

# cv2.imshow( 'BBB', BBB )
# cv2.waitKey()

delta = np.zeros_like(AAA)
delta[AAA == 0] =255


ret, thresh = cv2.threshold( delta, 0, 255, 0 )
im, contours, hi = cv2.findContours( thresh, 1, 2 )
for cnt in contours:
	approx = cv2.approxPolyDP( cnt, 0.02 * cv2.arcLength( cnt, True ), True )


kor =[]
for i in approx:
	kor.append(i[0])

plt.imshow(  delta )
plt.show()
#cv2.waitKey( )

y= kor[0]
y1= kor[1]
y2= kor[2]
y3= kor[3]

plt.imshow(  BBB[ y[1]:y2[1],y[0]:y2[0] ],"gray" )
plt.show()
#cv2.waitKey( )
cho_img = BBB[ y[1]:y2[1],y[0]:y2[0] ]

import cv2
import numpy as np



feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 14,
                       blockSize = 14 )


lk_params = dict( winSize  = (7,7),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.003))



cap = cv2.VideoCapture( 0 )
print( 'click "q" to stop' )

color = np.random.randint( 0, 255, (100, 3) )
ret, old_frame = cap.read( )
old_gray = cv2.cvtColor( old_frame, cv2.COLOR_BGR2GRAY )

k = np.zeros_like(old_gray)
k[ y[1]:y2[1],y[0]:y2[0] ] = cho_img

p0 = cv2.goodFeaturesToTrack( k, mask = None, **feature_params )

mask = np.zeros_like( old_frame )

while (1):


	ret, frame = cap.read( )
	frame_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
	p0 = cv2.goodFeaturesToTrack( k, mask = None, **feature_params )
	p1, st, err = cv2.calcOpticalFlowPyrLK( k, frame_gray, p0, None, **lk_params )

	good_new = p1[(st == 1)]
	good_old = p0[st == 1]

	for i, (new, old) in enumerate( zip( good_new, good_old ) ):
			a, b = new.ravel( )
			a1, b1 = old.ravel( )
			#mask = cv2.circle( frame, (a1, b1), 5, color[4].tolist( ), -1 )
			frame = cv2.circle( frame, (a, b), 5, color[i].tolist( ), 2 )

	img = cv2.add( frame, mask )

	x, y, w, h = cv2.boundingRect( good_new )

	cv2.rectangle( img, (x, y), (x + w, y + h), (0, 255, 0), 5 )
	#[y[1]: y2[1], y[0]: y2[0]]
	#cv2.rectangle( img, (y[1], y2[1]), (y[0], y2[0]), (0, 255, 0), 5 )

	cv2.namedWindow( 'image' )
	cv2.setMouseCallback( 'image', draw_circle )

	if cv2.waitKey( 1 ) & 0xff == ord( 'q' ):
		break
	old_gray = frame_gray.copy( )
	p0 = good_new.reshape( -1, 1, 2 )
	cv2.imshow( 'image', img )
	if (len( good_new ) > 500):

		ret, old_frame = cap.read( )
		old_gray = cv2.cvtColor( old_frame, cv2.COLOR_BGR2GRAY )

	else:
		p0 = cv2.goodFeaturesToTrack( old_gray, mask = None, **feature_params )

cv2.destroyAllWindows( )
cap.release( )
