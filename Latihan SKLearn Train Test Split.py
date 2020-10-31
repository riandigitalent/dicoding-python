# import library yang dibutuhkan
import sklearn
from sklearn import datasets

#import library untuk memangil funsi split  train  dan test 
from sklearn.model_selection import train_test_split
#loas iris datasey 
#sebuah dataset yang umum digunakan untuk masalah klasifikasi. Dataset ini memiliki jumlah 150 sampel
iris = datasets.load_iris()

#pisahkan atribut dan label pada iris dataset biar bisa di pake sama sklearn model datanya 
x = iris.data
y = iris.target
 
# membagi dataet menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#hitung panjang/ jumlah data pada x_test
len(x_test)
