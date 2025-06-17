import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# DATASETS
sex_offenders= pd.read_csv('datas/Sex_Offenders.csv')
crimes= pd.read_csv('datas/Crimes_2001_to_Present_Cut_Version.csv', low_memory= False)
#print(sex_offenders.head())
#print(sex_offenders.info())
#print(sex_offenders.columns)
#print(crimes.head())
#print(crimes.info())

# ---- TRATAMENTO DOS DADOS ---- #
sex_offenders['GENDER']= sex_offenders['GENDER'].astype('category')
sex_offenders['RACE']= sex_offenders['RACE'].astype('category')
sex_offenders['BIRTH DATE']= pd.to_datetime(sex_offenders['BIRTH DATE'])
sex_offenders['VICTIM MINOR']= sex_offenders['VICTIM MINOR'].astype('category')
sex_offenders['BLOCK'] = sex_offenders['BLOCK'].astype('category')
#
crimes['Date']= pd.to_datetime(crimes['Date'], format='%m/%d/%Y %I:%M:%S %p')
crimes ['Primary Type']= crimes['Primary Type'].astype ('category')
crimes ['Block']= crimes['Block'].astype('category')
crimes['IUCR']= crimes['IUCR'].astype('category')
crimes['Description']= crimes['Description'].astype('category')
crimes['Location Description']= crimes['Location Description'].astype('category')
crimes['Beat']= crimes['Beat'].astype('category')
crimes['District']= crimes['District'].astype('category')
crimes['Ward']= crimes['Ward'].astype('category')
crimes['Community Area']= crimes['Community Area'].astype('category')
crimes['FBI Code']= crimes['FBI Code'].astype('category')
crimes['Updated On']= pd.to_datetime(crimes['Updated On'], format='%m/%d/%Y %I:%M:%S %p') 
#print(crimes.info())
#print(crimes.columns)
#

#Cálculo da densidade de crimes sexuais pela densidade dos demais crimes por bloco
print(crimes['Primary Type'].unique())
tipos_sexuais = ['SEX OFFENSE', 'CRIM SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT'] 
agressao_sexual = crimes[crimes['Primary Type'].isin(tipos_sexuais)]
contagem_agressao = agressao_sexual.groupby('Block').size()
contagem_agressao = contagem_agressao[contagem_agressao > 0]
print (contagem_agressao)