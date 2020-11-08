#!/usr/bin/env python
# coding: utf-8

# In[3]:


# libraries 
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from math import *
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM, BinomialBayesMixedGLM 

## ols 
var=["Galumna","pa","totalabund","prop","SubsDens","WatrCont","Substrate","Shrub","Topo"]
mites=pd.read_csv("mites.csv",sep=",",header=0,names=var)
mites
mites.info()
#scatter matrix
scatter_matrix(mites, figsize=(12, 10), diagonal=var)
plt.show()
## recodage et plots 
WatrCont_presence=mites.WatrCont.loc[mites.pa==1]
WatrCont_absence=mites.WatrCont.loc[mites.pa==0]
d = {'absence':WatrCont_absence,'presence':WatrCont_presence}
df = pd.DataFrame(data=d)
#### plots
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1,3, wspace=0.5)
plt.figure(figsize=(10,5))
ax = plt.subplot(gs[0, 0])
plt.scatter(mites.WatrCont,mites.Galumna, c = 'red', marker = '.')
plt.ylabel('Abundance')
plt.xlabel('Water content')
ax = plt.subplot(gs[0, 1])
df.boxplot()
ax = plt.subplot(gs[0, 2])
plt.scatter(mites.WatrCont,mites.prop, c = 'red', marker = '.')
plt.ylabel('Proportion')
plt.xlabel('Water content')
plt.show()
###
mites.Galumna.shape
##ols
lm_abund=smf.ols('Galumna ~ WatrCont', data = mites).fit()
lm_abund.summary()
lm_pa=smf.ols('pa ~ WatrCont', data = mites).fit()
lm_pa.summary()
lm_prop=smf.ols('prop ~ WatrCont', data = mites).fit()
lm_prop.summary()
## population and regression plot
x = np.array(range(100,800))
y = lm_abund.params[1]*x+lm_abund.params[0]
plt.plot(x,y)
plt.ylabel('Galumna')
plt.xlabel('WatrCont')
plt.plot(mites.WatrCont,mites.Galumna,'o')
plt.show()
##partial regression plot
fig = sm.graphics.plot_partregress_grid(lm_abund)
fig.tight_layout(pad=1.0)
##risidual plot qq-normal plots
plt.subplot(121)
x1=np.array(lm_abund.fittedvalues)
y1=np.array(lm_abund.resid)
plt.scatter(x1, y1,s = 150, c = 'red', marker = '.')
plt.title('Rsisiduals vs Fitted')
plt.ylabel(' Residuals')
plt.xlabel('Fitted values')
plt.subplot(122)
residuals = np.array(lm_abund.resid)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot")
plt.show()
##sigma 
sigma_err = np.sqrt(lm_abund.scale)
sigma_err
##histogramme mites.galumna
plt.hist(mites.Galumna)
plt.title('histogramme of mites.Galumna')
plt.ylabel(' Frequency')
plt.xlabel('histogramme of mites.Galumna')
plt.show()
##another histogramme
plt.hist(mites.pa)
plt.title('histogramme of mites.pa')
plt.ylabel(' Frequency')
plt.xlabel('histogramme of mites.pa')
plt.show()
##glm binomial regression 
Topo=mites.Topo
Topo.loc[Topo=='Blanket']=0
Topo.loc[Topo=='Hummock']=1
Topo
d = {'mites.WatrCont': mites.WatrCont, 'Topo': Topo}
df = pd.DataFrame(data=d)
df
#sm.add_constant(X) to add the intercept
model_glm=sm.GLM(mites.pa,sm.add_constant(df).astype(float), data = mites, family =sm.families.Binomial()).fit()
model_glm.summary()
##  limitation 
CO2=pd.read_csv("CO2.csv",sep=" ",header=0)
CO2
model_CO2 =smf.ols('uptake ~ conc', data = CO2).fit()
model_CO2.summary()
##logistic regression and link function 
logit_reg =sm.GLM(mites.pa,sm.add_constant(df.astype(float)), data = mites, family =sm.families.Binomial(link=sm.families.links.logit)).fit()
logit_reg .summary()
### parametres of logistic regression exp
logit_reg.params
exp_logit_reg_coefficients=[exp(logit_reg.params[1]),exp(logit_reg.params[2])]#/ (1 - logit_reg.params[0]),logit_reg.params[1]/ (1 - logit_reg.params[1])]
print('exp(log(μ / (1 - μ)) = u / (1 - μ)')
print(exp_logit_reg_coefficients)
con_int_logit=pd.DataFrame(logit_reg.conf_int(0.25))
print('intervale de confiance')
#print(con_int_logit)
params = logit_reg.params
conf = logit_reg.conf_int(0.025)
conf.columns = ['2.5%', '97.5%']
print(np.exp(conf))
###GLM withcount data
names=['Unnamed: 0','UTM.EW','UTM.NS','Precipitation','Elevation','Age','Age.cat','Geology','Faramea_occidentalis']
faramea=pd.read_csv("faramea.csv",sep=",",header=0,names=names)
faramea.head(6)
##plot 
bins=np.arange(0, 46, step = 1)
plt.hist(faramea.Faramea_occidentalis, bins=bins)
plt.ylabel(' Frequency')
plt.xlabel('Number of Faramea_occidentalis')
plt.show()
## plot
plt.scatter(faramea.Elevation,faramea.Faramea_occidentalis, c = 'red', marker = '.')
plt.ylabel('Number of Faramea_occidentalis')
plt.xlabel('elevation(m)')
##poisson GLM
poisson_reg =sm.GLM(faramea.Faramea_occidentalis,sm.add_constant(faramea.Elevation.astype(float)), data = faramea, family =sm.families.Binomial()).fit()
poisson_reg.summary()
## negativebinomial glm
nbinomial_reg =sm.GLM(faramea.Faramea_occidentalis,sm.add_constant(faramea.Elevation.astype(float)), data = faramea, family =sm.families.NegativeBinomial()).fit()
nbinomial_reg.summary()
## GLMM
noms=['Unnamed: 0','reg','popu','gen','rack','nutrient','amd','status','total_fruits']
dat_tf=pd.read_csv("Banta_TotalFruits.csv",sep=",",header=0,names=noms)
dat_tf
##poisson glmm
#  poisson_glm=BinomialBayesMixedGLM.from_formula(dat_tf.total_fruits,dat_tf.nutrient*dat_tf.amd + dat_tf.rack + dat_tf.status +(1-dat_tf.popu)+(1-dat_tf.gen)).fit()

