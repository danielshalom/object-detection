import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('faces112x92.mat')

faces = data['faces']
target = data['target']

ny,nx = 112,92

plt.imshow(faces[1,:].reshape(ny,nx),interpolation='nearest',cmap='gray')
plt.show()
m, n = np.shape(faces)
def PCA(x, k):
	m, n = np.shape(x)
	ava_x = np.sum(x / m, axis=0)
	x = x - ava_x
	C = np.dot(x, x.T)

	evls, U = np.linalg.eig(C)

	sorted = np.argsort(evls)[::-1]
	U = U[:, sorted[:k]]
	new_U = np.dot(x.T, U)
	new_U = new_U / np.sum(new_U ** 2, axis=0)
	Z = np.dot(x, new_U)


	return new_U, Z, evls

new_U, Z, evls = PCA(faces, 10)

plt.plot(evls,'.-r')
plt.show()
plt.figure(figsize=(5, 2))
for i in np.arange(10):
	plt.subplot(2, 5, i + 1), plt.xticks([]), plt.yticks([])
	plt.imshow(new_U.T[i, :].reshape(ny, nx), interpolation='nearest', cmap='gray')
plt.show()
# #
plt.title("The projected 2-D data")
plt.scatter(Z[:, 0], Z[:, 1], 30, target.flatten(), cmap=plt.cm.rainbow)
plt.colorbar()
plt.grid()
plt.show()


for i in [5, 10, 50, 100, 200]:
 new_U, Z, evls = PCA(faces, i)

 new_U = new_U / np.sum(new_U ** 2, axis=0)

 img_U = np.dot(new_U, Z[1, :]) + np.sum(faces / m, axis=0)
 plt.title('k = %s'%i), plt.xticks([]), plt.yticks([])
 plt.imshow(img_U.reshape(ny, nx), interpolation='nearest', cmap='gray')

 plt.show()

from sklearn.svm import SVC

clf = SVC( C = 10000, kernel = 'rbf', gamma = 0.0001 )
errors_tst = []
errors_trn = []
best_k = []

for k in [2, 10, 20, 40, 46, 100, 200]:
	U, Z, evls = PCA( faces, k )
	
	y_80 = target[:int( len( target ) * 0.8 )]
	y_80 = y_80.ravel( )
	Z_tr = Z[:int( len( Z ) * 0.8 ), :]
	
	y_20 = target[int( len( target ) * 0.8 ):]
	Z_ts = Z[int( len( Z ) * 0.8 ):, :]
	
	clf.fit( Z_tr, y_80 )
	
	cnt_80 = clf.predict( Z_tr )
	
	Eror_80 = cnt_80 - y_80.T
	
	cnt_20 = clf.predict( Z_ts )
	
	Eror_20 = cnt_20 - y_20.T
	
	print( 'K =', k, ',', 'Eror_trn:', len( Eror_80[Eror_80 != 0] ), ',', 'Accuracy Training :',
	       100 - 100 * len( Eror_80[Eror_80 != 0] ) / (m * 0.8), '%', '\n' )
	
	print( 'K =', k, ',', 'Eror_tst:', len( Eror_20[Eror_20 != 0] ), ',', 'Accuracy test :',
	       100 - 100 * len( Eror_20[Eror_20 != 0] ) / (m * 0.2), '%', '\n' )
	print( '------' )
	#
	ers_tst = len( Eror_20[Eror_20 != 0] )
	ers_trn = len( Eror_80[Eror_80 != 0] )
	
	errors_tst.append( ers_tst )
	errors_trn.append( ers_trn )
	
	best_k.append( k )

plt.title( 'Training' ), plt.text( 0, 200, r'$ errors $' ), plt.text( 202, 0, ' K ' )
plt.plot( best_k, errors_trn, '.-r' )
plt.show( )

plt.title( 'Test' ), plt.text( 0, 60, r'$ errors $' ), plt.text( 200, 8, ' K ' )
plt.plot( best_k, errors_tst, '.-' )

plt.show( )
min_err = np.argmin( errors_tst )
print( 'The Best (test) K is', best_k[min_err], ','
       , 'Error :', np.min( errors_tst ), ',',
       'Accuracy :', 100 - 100 * np.min( errors_tst ) / (m * 0.2), '%' )


