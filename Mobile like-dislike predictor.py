
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss
import operator
import json
from IPython import display
import os
import warnings

np.random.seed(0)
warnings.filterwarnings("ignore")
THRESHOLD = 4


# read data from file
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv")

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


def data_clean(data):
    
    # Let's first remove all missing value features
    columns_to_remove = ['Also Known As','Applications','Audio Features','Bezel-less display'
                         'Browser','Build Material','Co-Processor','Browser'
                         'Display Colour','Mobile High-Definition Link(MHL)',
                         'Music', 'Email','Fingerprint Sensor Position',
                         'Games','HDMI','Heart Rate Monitor','IRIS Scanner', 
                         'Optical Image Stabilisation','Other Facilities',
                         'Phone Book','Physical Aperture','Quick Charging',
                         'Ring Tone','Ruggedness','SAR Value','SIM 3','SMS',
                         'Screen Protection','Screen to Body Ratio (claimed by the brand)',
                         'Sensor','Software Based Aperture', 'Special Features',
                         'Standby time','Stylus','TalkTime', 'USB Type-C',
                         'Video Player', 'Video Recording Features','Waterproof',
                         'Wireless Charging','USB OTG Support', 'Video Recording','Java']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    #Features having very low variance 
    columns_to_remove = ['Architecture','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Multivalued:
    columns_to_remove = ['Architecture','Launch Date','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE', 'Custom UI']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Not much important
    columns_to_remove = ['Bluetooth', 'Settings','Wi-Fi','Wi-Fi Features']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]
    
    return data


# In[5]:


train = data_clean(train)
test = data_clean(test)


# In[6]:


train = train[(train.isnull().sum(axis=1) <= 15)]

print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


def for_integer(test):
    try:
        test = test.strip()
        return int(test.split(' ')[0])
    except IOError:
           pass
    except ValueError:
        pass
    except:
        pass

def for_string(test):
    try:
        test = test.strip()
        return (test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass

def for_float(test):
    try:
        test = test.strip()
        return float(test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass
def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass

    
def for_Internal_Memory(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[1] == 'GB':
            return int(test[0])
        if test[1] == 'MB':
#             print("here")
            return (int(test[0]) * 0.001)
    except IOError:
           pass
    except ValueError:
        pass
    except:
        pass
    
def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


# In[9]:


def data_clean_2(x):
    data = x.copy()
    
    data['Capacity'] = data['Capacity'].apply(for_integer)

    data['Height'] = data['Height'].apply(for_float)
    data['Height'] = data['Height'].fillna(data['Height'].mean())

    data['Internal Memory'] = data['Internal Memory'].apply(for_Internal_Memory)

    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)

    data['Internal Memory'] = data['Internal Memory'].fillna(data['Internal Memory'].median())
    data['Internal Memory'] = data['Internal Memory'].astype(int)

    data['RAM'] = data['RAM'].apply(for_integer)
    data['RAM'] = data['RAM'].fillna(data['RAM'].median())
    data['RAM'] = data['RAM'].astype(int)

    data['Resolution'] = data['Resolution'].apply(for_integer)
    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())
    data['Resolution'] = data['Resolution'].astype(int)

    data['Screen Size'] = data['Screen Size'].apply(for_float)

    data['Thickness'] = data['Thickness'].apply(for_float)
    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())
    data['Thickness'] = data['Thickness'].round(2)

    data['Type'] = data['Type'].fillna('Li-Polymer')

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)

    data['Width'] = data['Width'].apply(for_float)
    data['Width'] = data['Width'].fillna(data['Width'].mean())
    data['Width'] = data['Width'].round(2)

    data['Flash'][data['Flash'].isna() == True] = "Other"

    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"

    data['Num_cores'] = data['Processor'].apply(for_string)
    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"


    data['Processor_frequency'] = data['Processor'].apply(find_freq)
    #because there is one entry with 208MHz values, to convert it to GHz
    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208
    data['Processor_frequency'] = data['Processor_frequency'].fillna(data['Processor_frequency'].mean())
    data['Processor_frequency'] = data['Processor_frequency'].round(2)

    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"

    #simplifyig Operating System to os_name for simplicity
    data['os_name'] = data['Operating System'].apply(for_string)
    data['os_name'][data['os_name'].isna() == True] = "Other"

    data['Sim1'] = data['SIM 1'].apply(for_string)

    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"

    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"

    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"

    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"

    data['Weight'] = data['Weight'].apply(for_integer)
    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())
    data['Weight'] = data['Weight'].astype(int)

    data['SIM 2'] = data['SIM 2'].apply(for_string)
    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"
    
    return data


# In[10]:


train = data_clean_2(train)
test = data_clean_2(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[11]:


def data_clean_3(x):
    
    data = x.copy()

    columns_to_remove = ['User Available Storage','SIM Size','Chipset','Processor','Autofocus','Aspect Ratio','Touch Screen',
                        'Bezel-less display','Operating System','SIM 1','USB Connectivity','Other Sensors','Graphics','FM Radio',
                        'NFC','Shooting Modes','Browser','Display Colour' ]

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]


    columns_to_remove = [ 'Screen Resolution','User Replaceable','Camera Features',
                        'Thickness', 'Display Type']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]


    columns_to_remove = ['Fingerprint Sensor', 'Flash', 'Rating Count', 'Review Count','Image Resolution','Type','Expandable Memory',                        'Colours','Width','Model']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    return data


# In[12]:


train = data_clean_3(train)
test = data_clean_3(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# In[13]:


# one hot encoding

train_ids = train['PhoneId']
test_ids = test['PhoneId']

cols = list(test.columns)
cols.remove('PhoneId')
cols.insert(0, 'PhoneId')

combined = pd.concat([train.drop('Rating', axis=1)[cols], test[cols]])
print(combined.shape)
print(combined.columns)

combined = pd.get_dummies(combined)
print(combined.shape)
print(combined.columns)

train_new = combined[combined['PhoneId'].isin(train_ids)]
test_new = combined[combined['PhoneId'].isin(test_ids)]


# In[14]:


train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')


# In[15]:


# check the number of features and data points in train
print("Number of data points in train: %d" % train_new.shape[0])
print("Number of features in train: %d" % train_new.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test_new.shape[0])
print("Number of features in test: %d" % test_new.shape[1])


# In[16]:


train_new.head()


# In[17]:


test_new.head()


# # Dummy Solution

# In[18]:


submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':[1]*test_new.shape[0]})
submission = submission[['PhoneId', 'Class']]
submission.head()


# In[19]:


submission.to_csv("submission.csv", index=False)


# In[20]:


mod_ram = []
for i in train_new['RAM']:
    if(i > 8):
        i = i/1000
        mod_ram.append(i)
    else:
        mod_ram.append(i)


# In[21]:


train_new['RAM'] = mod_ram


# In[22]:


train_new.iloc[74, 10] = 4


# In[23]:


train_new.iloc[126, 6] = 0.016


# In[24]:


phoneid = train_new['PhoneId']


# In[25]:


bin_rating = train_new['Rating'].map(lambda x: 0 if x<4 else 1)


# In[26]:


del train_new['Rating']


# In[27]:


train_new['Rating'] = bin_rating


# In[28]:


del train_new['PhoneId']


# In[29]:


max_ = min_ = 0
for i in train_new['Height']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Height']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 0] = (i - min_) / diff
    c += 1


# In[30]:


max_ = min_ = 0
for i in test_new['Height']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Height']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 1] = (i - min_) / diff
    c += 1


# In[31]:


max_ = min_ = 0
for i in train_new['Screen Size']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Screen Size']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 1] = (i - min_) / diff
    c += 1


# In[32]:


max_ = min_ = 0
for i in test_new['Screen Size']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Screen Size']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 2] = (i - min_) / diff
    c += 1


# In[33]:


max_ = min_ = 0
for i in train_new['RAM']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['RAM']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 2] = (i - min_) / diff
    c += 1


# In[34]:


max_ = min_ = 0
for i in test_new['RAM']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['RAM']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 3] = (i - min_) / diff
    c += 1


# In[35]:


max_ = min_ = 0
for i in train_new['Internal Memory']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Internal Memory']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 3] = (i - min_) / diff
    c += 1


# In[36]:


max_ = min_ = 0
for i in test_new['Internal Memory']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Internal Memory']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 4] = (i - min_) / diff
    c += 1


# In[37]:


max_ = min_ = 0
for i in train_new['Pixel Density']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Pixel Density']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 4] = (i - min_) / diff
    c += 1


# In[38]:


max_ = min_ = 0
for i in test_new['Pixel Density']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Pixel Density']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 5] = (i - min_) / diff
    c += 1


# In[39]:


max_ = min_ = 0
for i in train_new['Weight']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Weight']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 5] = (i - min_) / diff
    c += 1


# In[40]:


max_ = min_ = 0
for i in test_new['Weight']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Weight']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 6] = (i - min_) / diff
    c += 1


# In[41]:


max_ = min_ = 0
for i in train_new['Screen to Body Ratio (calculated)']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Screen to Body Ratio (calculated)']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 6] = (i - min_) / diff
    c += 1


# In[42]:


max_ = min_ = 0
for i in test_new['Screen to Body Ratio (calculated)']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Screen to Body Ratio (calculated)']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 7] = (i - min_) / diff
    c += 1


# In[43]:


max_ = min_ = 0
for i in train_new['Resolution']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Resolution']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 7] = (i - min_) / diff
    c += 1


# In[44]:


max_ = min_ = 0
for i in test_new['Resolution']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Resolution']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 8] = (i - min_) / diff
    c += 1


# In[45]:


max_ = min_ = 0
for i in train_new['Capacity']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Capacity']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 8] = (i - min_) / diff
    c += 1


# In[46]:


max_ = min_ = 0
for i in test_new['Capacity']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Capacity']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 9] = (i - min_) / diff
    c += 1


# In[47]:


max_ = min_ = 0
for i in train_new['Processor_frequency']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in train_new['Processor_frequency']:
#     print(train_new.iloc[c, 0])
    train_new.iloc[c, 9] = (i - min_) / diff
    c += 1


# In[48]:


max_ = min_ = 0
for i in test_new['Processor_frequency']:
    if(i > max_):
        max_ = i
    if(i < min_):
        min_ = i
diff = max_ - min_
c = 0
print(max_, min_, sep = '\n')
for i in test_new['Processor_frequency']:
#     print(train_new.iloc[c, 0])
    test_new.iloc[c, 10] = (i - min_) / diff
    c += 1


# In[49]:


X = train_new.drop('Rating', axis = 1)
Y = train_new['Rating']


# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)


# In[51]:


X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values


# In[52]:


class Perceptron:
  
  def __init__(self):
    self.w = 0
    self.b = 0
    
  def model(self, x):
    return 1 if np.dot(self.w, x) >= self.b else 0
  
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  def fit(self, X, Y, epochs = 1, lr = 1):
    self.w = np.ones(X.shape[1])
    self.b = 0
    accuracy = {}
    max_accuracy = 0
    
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if(y == 1 and y_pred == 0):
          self.w += x * lr
          self.b += 1 * lr
        elif(y == 0 and y_pred == 1):
          self.w -= x * lr
          self.b -= 1 * lr
      accuracy[i] = accuracy_score(self.predict(X), Y)
      if(accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.w
        chkptb = self.b
    
    self.b = chkptb
    self.w = chkptw
    print(max_accuracy)
    
perceptron = Perceptron()


perceptron.fit(X_test, Y_test, 38000, 0.00003)
