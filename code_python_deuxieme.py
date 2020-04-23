import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#-----------------------------------------------------
#Lecture des données
print("Loading data---------------------")

df1 = pd.read_csv('diet.csv', index_col=0)
df2 = pd.read_csv('examination.csv', index_col=0)

print("df1 shape : ",df1.shape)
print("df2 shape : ",df2.shape)

print("df1 head")
print(df1.head(3))
print("df2 head")
print(df2.head(3))

# check if df shape is correct
assert df1.shape == (9813, 167)
assert df2.shape == (9813, 223)

#------------------------------------------------------------------------------------------------------
#Colonne à supprimer
print("Droping columns---------------------")

diet_to_drop = [
  'WTDRD1' , 'WTDR2D' , 'DR1DRSTZ' , 'DR1EXMER' , 'DRABF' , 'DRDINT' , 'DR1DBIH' , 'DR1DAY' , 'DR1LANG' , 'DR1MNRSP' , 'DR1HELPD' , 'DBQ095Z' , 'DRQSPREP' , 'DR1STY' , 'DR1SKY' , 'DRQSDIET' , 'DRQSDT1' , 'DRQSDT2' , 'DRQSDT3' , 'DRQSDT4' , 'DRQSDT5' , 'DRQSDT6' , 'DRQSDT7' , 'DRQSDT8' , 'DRQSDT9' , 'DRQSDT10' , 'DRQSDT11' , 'DRQSDT12' , 'DRQSDT91' , 'DR1TNUMF' , 'DR1TPROT' , 'DR1TCARB' , 'DR1TSUGR' , 'DR1TFIBE' , 'DR1TTFAT' , 'DR1TSFAT' , 'DR1TMFAT' , 'DR1TPFAT' , 'DR1TCHOL' , 'DR1TATOC' , 'DR1TATOA' , 'DR1TRET' , 'DR1TVARA' , 'DR1TACAR' , 'DR1TBCAR' , 'DR1TCRYP' , 'DR1TLYCO' , 'DR1TLZ' , 'DR1TVB1' , 'DR1TVB2' , 'DR1TNIAC' , 'DR1TVB6' , 'DR1TFOLA' , 'DR1TFA' , 'DR1TFF' , 'DR1TFDFE' , 'DR1TCHL' , 'DR1TVB12' , 'DR1TB12A' , 'DR1TVC' , 'DR1TVD' , 'DR1TVK' , 'DR1TCALC' , 'DR1TPHOS' , 'DR1TMAGN' , 'DR1TIRON' , 'DR1TZINC' , 'DR1TCOPP' , 'DR1TSODI' , 'DR1TPOTA' , 'DR1TSELE' , 'DR1TCAFF' , 'DR1TTHEO' , 'DR1TMOIS' , 'DR1TS040' , 'DR1TS060' , 'DR1TS080' , 'DR1TS100' , 'DR1TS120' , 'DR1TS140' , 'DR1TS160' , 'DR1TS180' , 'DR1TM161' , 'DR1TM181' , 'DR1TM201' , 'DR1TM221' , 'DR1TP182' , 'DR1TP183' , 'DR1TP184' , 'DR1TP204' , 'DR1TP205' , 'DR1TP225' , 'DR1TP226' , 'DR1.300' , 'DR1.320Z' , 'DR1.330Z' , 'DR1BWATZ' , 'DR1TWS' , 'DRD340' , 'DRD350A' , 'DRD350AQ' , 'DRD350B' , 'DRD350BQ' , 'DRD350C' , 'DRD350CQ' , 'DRD350D' , 'DRD350DQ' , 'DRD350E' , 'DRD350EQ' , 'DRD350F' , 'DRD350FQ' , 'DRD350G' , 'DRD350GQ' , 'DRD350H' , 'DRD350HQ' , 'DRD350I' , 'DRD350IQ' , 'DRD350J' , 'DRD350JQ' , 'DRD350K' , 'DRD360' , 'DRD370A' , 'DRD370AQ' , 'DRD370B' , 'DRD370BQ' , 'DRD370C' , 'DRD370CQ' , 'DRD370D' , 'DRD370DQ' , 'DRD370E' , 'DRD370EQ' , 'DRD370F' , 'DRD370FQ' , 'DRD370G' , 'DRD370GQ' , 'DRD370H' , 'DRD370HQ' , 'DRD370I' , 'DRD370IQ' , 'DRD370J' , 'DRD370JQ' , 'DRD370K' , 'DRD370KQ' , 'DRD370L' , 'DRD370LQ' , 'DRD370M' , 'DRD370MQ' , 'DRD370N' , 'DRD370NQ' , 'DRD370O' , 'DRD370OQ' , 'DRD370P' , 'DRD370PQ' , 'DRD370Q' , 'DRD370QQ' , 'DRD370R' , 'DRD370RQ' , 'DRD370S' , 'DRD370SQ' , 'DRD370T' , 'DRD370TQ' , 'DRD370U' , 'DRD370UQ' , 'DRD370V'
]

examination_to_drop = [
  'OHX01TC' , 'OHX02TC' , 'BMDBMIC' , 'CSQ450' ,'PEASCST1' , 'PEASCTM1' , 'PEASCCT1' , 'BPXCHR' , 'BPAARM' , 'BPACSZ' , 'BPXPLS' , 'BPXPULS' , 'BPXPTY' , 'BPXML1' , 'BPXSY1' , 'BPXDI1' , 'BPAEN1' , 'BPXSY2' , 'BPXDI2' , 'BPAEN2' , 'BPXSY3' , 'BPXDI3' , 'BPAEN3' , 'BPXSY4' , 'BPXDI4' , 'BPAEN4' , 'BMDSTATS' , 'BMIWT' , 'BMXRECUM' , 'BMIRECUM' , 'BMXHEAD' , 'BMIHEAD' , 'BMIHT', 'BMXLEG' , 'BMILEG' , 'BMXARML' , 'BMIARML' , 'BMXARMC' , 'BMIARMC' , 'BMXWAIST' , 'BMIWAIST' , 'BMXSAD1' , 'BMXSAD2' , 'BMXSAD3' , 'BMXSAD4' , 'BMDAVSAD' , 'BMDSADCM' , 'MGDEXSTS' , 'MGD050' , 'MGD060' , 'MGQ070' , 'MGQ080' , 'MGQ090' , 'MGQ100' , 'MGQ110' , 'MGQ120' , 'MGD130' , 'MGQ90DG' , 'MGDSEAT' , 'MGAPHAND' , 'MGATHAND' , 'MGXH1T1' , 'MGXH1T1E' , 'MGXH2T1' , 'MGXH2T1E' , 'MGXH1T2' , 'MGXH1T2E' , 'MGXH2T2' , 'MGXH2T2E' , 'MGXH1T3' , 'MGXH1T3E' , 'MGXH2T3' , 'MGXH2T3E' , 'MGDCGSZ' , 'OHDEXSTS' , 'OHXIMP' , 'OHX03TC' , 'OHX04TC' , 'OHX05TC' , 'OHX06TC' , 'OHX07TC' , 'OHX08TC' , 'OHX09TC' , 'OHX10TC' , 'OHX11TC' , 'OHX12TC' , 'OHX13TC' , 'OHX14TC' , 'OHX15TC' , 'OHX16TC' , 'OHX17TC' , 'OHX18TC' , 'OHX19TC' , 'OHX20TC' , 'OHX21TC' , 'OHX22TC' , 'OHX23TC' , 'OHX24TC' , 'OHX25TC' , 'OHX26TC' , 'OHX27TC' , 'OHX28TC' , 'OHX29TC' , 'OHX30TC' , 'OHX31TC' , 'OHX32TC' , 'OHX02CTC' , 'OHX03CTC' , 'OHX04CTC' , 'OHX05CTC' , 'OHX06CTC' , 'OHX07CTC' , 'OHX08CTC' , 'OHX09CTC' , 'OHX10CTC' , 'OHX11CTC' , 'OHX12CTC' , 'OHX13CTC' , 'OHX14CTC' , 'OHX15CTC' , 'OHX18CTC' , 'OHX19CTC' , 'OHX20CTC' , 'OHX21CTC' , 'OHX22CTC' , 'OHX23CTC' , 'OHX24CTC' , 'OHX25CTC' , 'OHX26CTC' , 'OHX27CTC' , 'OHX28CTC' , 'OHX29CTC' , 'OHX30CTC' , 'OHX31CTC' , 'OHX02CSC' , 'OHX03CSC' , 'OHX04CSC' , 'OHX05CSC' , 'OHX06CSC' , 'OHX07CSC' , 'OHX08CSC' , 'OHX09CSC' , 'OHX10CSC' , 'OHX11CSC' , 'OHX12CSC' , 'OHX13CSC' , 'OHX14CSC' , 'OHX15CSC' , 'OHX18CSC' , 'OHX19CSC' , 'OHX20CSC' , 'OHX21CSC' , 'OHX22CSC' , 'OHX23CSC' , 'OHX24CSC' , 'OHX25CSC' , 'OHX26CSC' , 'OHX27CSC' , 'OHX28CSC' , 'OHX29CSC' , 'OHX30CSC' , 'OHX31CSC' , 'OHX02SE' , 'OHX03SE' , 'OHX04SE' , 'OHX05SE' , 'OHX07SE' , 'OHX10SE' , 'OHX12SE' , 'OHX13SE' , 'OHX14SE' , 'OHX15SE' , 'OHX18SE' , 'OHX19SE' , 'OHX20SE' , 'OHX21SE' , 'OHX28SE' , 'OHX29SE' , 'OHX30SE' , 'OHX31SE' , 'CSXEXCMT' , 'CSQ245' , 'CSQ260A' , 'CSQ260D' , 'CSQ260G' , 'CSQ260I' , 'CSQ260N' , 'CSQ260M' , 'CSQ270' , 'CSQ460' , 'CSQ470' , 'CSQ480' , 'CSQ490' , 'CSXQUIPG' , 'CSXQUIPT' , 'CSXNAPG' , 'CSXNAPT' , 'CSXQUISG' , 'CSXQUIST' , 'CSXSLTSG' , 'CSXSLTST' , 'CSXNASG' , 'CSXNAST' , 'CSXTSEQ' , 'CSXCHOOD' , 'CSXSBOD' , 'CSXSMKOD' , 'CSXLEAOD' , 'CSXSOAOD' , 'CSXGRAOD' , 'CSXONOD' , 'CSXNGSOD' , 'CSXSLTRT' , 'CSXSLTRG' , 'CSXNART' , 'CSXNARG' , 'CSAEFFRT'
]

df1.drop(diet_to_drop, axis = 1, inplace = True)
print("df1 shape : ",df1.shape)

df2.drop(examination_to_drop, axis = 1, inplace = True)
print("df2 shape : ",df2.shape)

#------------------------------------------------------------------------------------------
#Rename les colonnes
print("Rename columns---------------------")

#Rename columns
df1.rename(columns={"SEQN" : "id", "DBD100" : "salt", "DR1TKCAL" : "calories" , "DR1TALCO" : "alcohol" }, inplace=True)
print("df1 head")
print(df1.head(3))

#Rename columns
df2.rename(columns={"SEQN" : "id", "BMXWT" : "weight", "BMXHT" : "height" , "BMXBMI" : "mass_index", "OHDDESTS" : "dentition", "CSXEXSTS" : "smell_taste_perf", "CSQ241" : "pregnant_breastfeed"}, inplace=True)
print("df2 head")
print(df2.head(3))

#------------------------------------------------------------------------------------------
#Fusionne les df
print("Merge dfs---------------------")

print("Merge frame to create df")
#Merge frame
frames = [df1, df2]
df = pd.concat(frames, axis=1, join='inner', ignore_index=False)
print("df shape : ",df.shape)

#----------------------------------------------------
#Remplir les valeurs manquantes
print("Fill empty value of df---------------------")

df.dropna(subset=['height', 'weight'], inplace = True)
df["pregnant_breastfeed"].fillna(1 , inplace=True) #If pregnant
df["salt"].fillna(df["salt"].mean(), inplace=True) 
df["calories"].fillna(df["calories"].mean() ,inplace=True) 
df["alcohol"].fillna(df["alcohol"].mean() ,inplace=True) 
df["mass_index"].fillna(df["mass_index"].mean(),inplace=True) 
df["dentition"].fillna(df["dentition"].mean() ,inplace=True) 
df["smell_taste_perf"].fillna(df["smell_taste_perf"].mean() , inplace=True) 

print("df head")
print(df.head(3))

#-----------------------------------------------------------------------
#Change les phrases en valeur pour être modifié en int par la suite
print("Change df type to int---------------------")

df = df.astype(int)
print("df head")
print(df.head(3))


#--------------------------------------------------------------------
#Préparation de ce que l'on veut prédire

y = df.weight
X = df.drop({'weight'}, axis=1, inplace=False)
y.hist(bins=150)

#--------------------------------------------------
print("Applying linear regression---------------------")

print("X shape : ",X.shape)
print("y shape : ",y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
print("train set shape: ", X_train.shape, y_train.shape)
print("test set shape: ", X_test.shape, y_test.shape)

print("linear regression")
reg = LinearRegression()
reg.fit(X_train, y_train)
train_score = reg.score(X_train, y_train)
test_score=reg.score(X_test, y_test)
print('train score =', train_score)
print('test score = {}'.format(test_score))

print("Predict a known weigh")
index_to_predict = 100
print("Value to predict is ",y[73557+index_to_predict])
print(reg.predict(np.array(X.iloc[index_to_predict]).reshape(1,-1))[0])

#--------------------------------------------------
#Le poids de l'utilisateur à prédir en fonction de ce qu'il rentre
print("Try to predict user weight---------------------")

#List of Tuples
values = [(2, 1574, 0, 172, 26, 1, 1, 1)]
#Create a DataFrame object
newdf = pd.DataFrame(values, columns = ['salt' , 'calories', 'alcohol', 'height' , 'mass_index', 'dentition' , 'smell_taste_perf','pregnant_breastfeed'], index=[0]) 

print("newdf head")
print(newdf.head(1))

X = newdf
print("Value to predict is an unknown weight")
print("Value predicted is : ",reg.predict(np.array(X.iloc[0]).reshape(1,-1))[0])
