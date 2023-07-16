
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models,pull,save_model




with st.sidebar:
    st.title("Caffeine Basic EDA")
    choises = st.radio("menü",["veri setini tanıma","içecekler","grafikler","autoprofile","ml","ml2"])
    st.info("Gaye Çetindere")

df= pd.read_csv('caffeine.csv')

if choises == "veri setini tanıma":
   st.title("dataframe")
   st.write(df)
   st.info("verimizin sütunları")
   st.write(df.columns)
   st.info("describe metodu bizim verimizi istatiksel olarak anlamamızı sağlar")
   st.write(df.describe())
   st.info("null değerlerimiz var mı?")

   st.write(df.isnull().sum())
   st.info("sütunlarımızın veri tipleri")
   st.write(df.dtypes)
   st.info("nununique metodu verimizin tek olan değerlerini yani unique olan değerlerini sayar ve tekrarlamayan verilerin sayısını bize gösterir")
   st.write(df.nunique())
   st.info("corr metodu verimizin korelasyon ilişkisini gösterir")
   st.write(df.corr())
   st.info("kafein oranına göre en yüksekten başlayark sıraladığımızda verimiz:")
   new = df.sort_values(by=['Caffeine (mg)'], inplace=False, ascending=False)
   st.write(new)
  







if choises == "içecekler":
   st.info("verimizde kaç içecek türü var?")
   st.write(df["type"].value_counts())
   st.info("hacim başı kafein oraını bulmak için verimizi önce kafein oranına göre yüksekten başlayarak sıralıyoruz ve kafeinleri hacimlere böldük")
   new = df.sort_values(by=['Caffeine (mg)'], inplace=False, ascending=False)
   new["per volume caffeine"] = new["Caffeine (mg)"]/new["Volume (ml)"]
   st.write(new)

   gk=df.groupby('type')

   if st.checkbox("tüm kahveleri göster"):
      st.title("tüm kahveler")
    
      coffe = gk.get_group('Coffee')
      st.write(coffe)
      st.info("sirasiyla kahvenin kalorisi,kaffeini ve miktarinin ortalaması")
      st.write(coffe['Calories'].mean())
      st.write(coffe['Caffeine (mg)'].mean())
      st.write(coffe['Volume (ml)'].mean())
    


   if st.checkbox("tüm yumuşak içecekleri göster"):
      st.title("tüm yumuşak içecekler")

      soft = gk.get_group('Soft Drinks')
      st.write(soft)
      st.info("sirasiyla yumuşak içeceklerin kalorisi,kaffeini ve miktarinin ortalaması")
      st.write(soft['Calories'].mean())
      st.write(soft['Caffeine (mg)'].mean())
      st.write(soft['Volume (ml)'].mean())

    
   if st.checkbox("tüm enerji içeceklerini göster"):
    st.title("tüm enerji içecekleri ")
    energy = gk.get_group('Energy Drinks')
    st.write(energy)

    st.info("sirasiyla enerji içeceklerinin kalorisi,kafeini ve miktarinin ortalaması")
    st.write(energy['Calories'].mean())
    st.write(energy['Caffeine (mg)'].mean())
    st.write(energy['Volume (ml)'].mean())

     
   if st.checkbox("tüm çaylari göster"):
    st.title("tüm çaylar")
    tea = gk.get_group('Tea')
    st.write(tea)
    st.info("sirasiyla çayların kalorisi,kafeini ve miktarinin ortalaması")
    st.write(tea['Calories'].mean())
    st.write(tea['Caffeine (mg)'].mean())
    st.write(tea['Volume (ml)'].mean())

   if st.checkbox("sıfır kalorilileri göster"):
     st.title("sıfır kalorililer")
     zc=df.groupby('Calories')
     zero = zc.get_group(0)
     st.write(zero) 
     st.write(zero["type"].value_counts())

if choises == "grafikler":
  
  
  gk=df.groupby('type')
  coffe = gk.get_group('Coffee')
  soft = gk.get_group('Soft Drinks')
  energy = gk.get_group('Energy Drinks')
  tea = gk.get_group('Tea')
  st.title("karşılaştırma")
  a = coffe["Calories"].mean()
  b = coffe["Caffeine (mg)"].mean()
  c = coffe["Volume (ml)"].mean()
  d = energy["Calories"].mean()
  e = energy["Caffeine (mg)"].mean()
  f = energy["Volume (ml)"].mean()
  g = soft["Calories"].mean()
  h = soft["Caffeine (mg)"].mean()
  j = soft["Volume (ml)"].mean()
  k = tea["Calories"].mean()
  l = tea["Caffeine (mg)"].mean()
  m = tea["Volume (ml)"].mean()
  X = ['calories','caffeine','volume']
  coff = [a,b,c]
  energ = [d,e,f]
  so = [g,h,j]
  te = [k,l,m]

  color1 = '#B35E5E'
  color2 = '#5E74B3'
  color3 = '#5E94B3'

  w = 0.2

  bar1 = np.arange(len(X))
  bar2 = [i + w for i in bar1] 
  bar3 = [i + w for i in bar2]
  bar4= [i+w for i in bar3]

  plt.figure(figsize=(10,10), dpi= 80)

  plt.bar(bar1, coff, w, label = 'coffee', color = color1)
  plt.bar(bar2, energ, w, label = 'energy drinks',color = color2)
  plt.bar(bar3,so,w,label = 'soft drinks',color = color3)
  plt.bar(bar4,te,w,label = 'tea')

  

  plt.xlabel("categories")
  plt.ylabel("mean values")
  plt.title("comparison")
  plt.xticks(bar1+0.3,X)

  plt.legend()

  st.pyplot()

  ###
    
  st.title("sıfır kaloriler")
  ab = df.groupby('Calories')
  zero = ab.get_group(0)
  zero["type"].value_counts()
  y = zero["type"].value_counts()
  mylabels = ["coffee","energy drinks","tea","soft drinks","water","energy shots"]
  my_colors = ["#B36C5E","#D87020","#E5A309","#C9970E","#C9C90E","#CEE21A"]
  plt.figure(figsize=(10,10), dpi= 80)

  st.write(plt.pie(y, labels=mylabels,colors= my_colors))
  st.pyplot()

  st.title("bar plot")
  
  st.write(df.plot(kind='bar'))
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()

    ##
  st.title("en yüksek kafein oranına sahip 10 içecek")
  color4 = '#ABB35E'
  new = df.sort_values(by=['Caffeine (mg)'], inplace=False, ascending=False)
  new["per volume caffeine"] = new["Caffeine (mg)"]/new["Volume (ml)"]
  name = new["drink"].head(10)
  caff = new["per volume caffeine"].head(10)
  plt.figure(figsize=(10,10), dpi= 80)
  plt.bar(name[0:10], caff[0:10],0.4,color=color4)
  plt.xticks(rotation=70, horizontalalignment='right') 
  st.pyplot() 

  ##

 

  st.title("Aykırı veriler")
  out =df.select_dtypes(include=['float64','int64'])
  for i in out:
    l = out[i]
    st.info(i)
    plt.boxplot(x=l)
    st.pyplot()

if choises == "autoprofile":
  st.title("automated eda")
  profile_report = df.profile_report()
  st_profile_report(profile_report)
  
if choises=="ml":
  st.title("aouto ml")
  target = st.selectbox("select",df.columns)
  setup(df,target=target)
  setup_df = pull()
  best_model = compare_models()
  compare_df=pull()
  st.dataframe(compare_df)
  best_model
if choises=="ml2":
  st.title("aouto ml")
  
  setup(df,target=df['type'])
  setup_df = pull()
  best_model = compare_models()
  compare_df=pull()
  st.dataframe(compare_df)
  best_model



     
 