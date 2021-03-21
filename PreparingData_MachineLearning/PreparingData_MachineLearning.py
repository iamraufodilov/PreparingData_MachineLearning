# Loading data
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
path = r'G:\rauf\STEPBYSTEP\Data\pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
array = data.values
print("actual data is looks like: \n", array[0:3])

# Scaling
MMScal = MinMaxScaler()
MMScal.fit(array)
scaled_data = MMScal.transform(array)
set_printoptions(precision=2)
print("our scaled data is: \n", scaled_data[:3])

# Normalization

#L1 normalisation
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
DNorm = Normalizer(norm='l1')
DNorm.fit(array)
data_normalized = DNorm.transform(array)
set_printoptions(precision=2)
print("normalized data with l1: \n", data_normalized[0:3])


#L2 normalization
L2DNorm = Normalizer(norm='l2')
L2DNorm.fit(array)
L2data_normalized = L2DNorm.transform(array)
set_printoptions(precision=2)
print("normalized data with l2: \n", L2data_normalized[0:3])

# Binarization
from sklearn.preprocessing import Binarizer
my_binarizer = Binarizer(threshold=0.5)
my_binarizer.fit(array)
binarized_data = my_binarizer.transform(array)
print("our binarized data is: \n", binarized_data[0:3])

# Standarization
from sklearn.preprocessing import StandardScaler
DScal = StandardScaler()
DScal.fit(array)
data_standarted = DScal.transform(array)
set_printoptions(precision=2)
print("our standarized data is: \n", data_standarted[0:3])

# Labelling Encoding
from sklearn.preprocessing import LabelEncoder
input_labels = ['red','black','red','green','black','yellow','white']
LEnc = LabelEncoder()
LEnc.fit(input_labels)
test_labels = ['green','red','black']
encoded_test_labels = LEnc.transform(test_labels)
print("our actual test labels: \n", test_labels)
print("our encoded test labels are: \n", encoded_test_labels)
# reverse encoding
encoded_values = [3,0,4,1]
decoded_encoded_values = LEnc.inverse_transform(encoded_values)
print("our actual given encoded values are: \n", encoded_values)
print("reverse transformed result: \n", decoded_encoded_values)
