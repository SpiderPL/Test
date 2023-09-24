!pip install pyarrow
df1= pd.read_parquet('// X.parquet')
df1.to_parquet('x.parquet')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important: } </style>"))

import matplotlib as plt
plt.rcParams['figure.figsize'] =(30,8)
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

means2 = pd.DataFrame(columns=['Datetime_local','A','B'])
frame = pd.DataFrame(data=pd.date_range(start='2023-09-10', end='2023-10-21', freq='1h'),columns=['Datetime_local'])
df=pd.concat([df,df1])
df=df.merge(df1, left_on='Datetime_local', right_on='Datetime_local', how='left')
df['Datetime_local']=pd.to_datetime(df['Datetime_local'], format'%Y-%m-%d %H:%H:%S', errors='coerce')

pd.Timedelta(hourss=1)
.drop_duplicates(subset=['key'], keep='last')
df[['start_date','End_date']]=df.Hours.str.split("-",expand=True)
.replace('-',0,inplace=True)
dfx[name]=dfx[name].str.replace(',','.').astype(float)
dfx[name]=dfx[name].astype(float)
df['H'].apply(lambda x: x[0:4] + '-' + x[4:6])
df.to_frame()
df.columns=['Data',1,2,3,4,5,6,7,8]
df=df[df['A'].notna()]
df=df[df['A'].str.contains('D9') == False]
df['pos']=df['pos'].dt.total_seconds()/3600
.shift(periods=-1,fill_value=0)
df.loc[(df['A']=='----'),'A']=-1

from os import walk
dir_path=r'C:\Projekty'
list_of_files = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    for name in file_names:
        link_to_file= str(''.join(dir_path) + '\\' +str(name))
        print(link_to_file)
        list_of_files.append(name)



import os
folder_name
path= os.getcwd() +'\\' + str(folder_name)
isExist=os.path.exists(path)
if not isExist:
    os.makedirs(path)

string_list = [str(element) for element in df[df.columns[0]].unique().tolist()]

import unicodedata
list_A.append(unicodedata.normalize("NFKD",name))
list1= list(set(list1))
list1= list(filter(lambda i: len(str(i))>4, list1))
dfx['Datetime_local']=pd.to_datetime(df['index'], errors='ignore', unit='h', origin=list1[k].strftime(%Y-%m-%d %X))
dfx['Datetime_local']=pd.to_datetime(df['index'].str.strip(), format='%Y-%m-%d')
rename(columns={'A':'B', inplace=True})
.extend
.astype('string')


!pip install bokeh
from bokeh.plotting import figure, output_file, show,save, ColumnDataSource
from bokeh.models import SingleIntervalTicker, LinearAxis, BoxAnnotation

frame =df11.copy()
TITLE='Wykres pomocniczy'
output_file(TITLE + ".html")
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save,xpan,ypan"
#p = figure(plot_width=1800, plot_height=1000, tools=TOOLS, toolbar_location='above', title=TITLE, x_axis_type='datetime')
p = figure(width=1800, height=1000, tools=TOOLS, toolbar_location='above', title=TITLE, x_axis_type='linear')
p.toolbar.logo = 'grey'
p.background_fill_color = '#dddddd'
p.xaxis.axis_label='Datetime'
p.yaxis.axis_label='Data'
#ind=frame['Datetime_local']
ind=frame['index']
p.line(x=ind, y=frame['a'].values, legend_label='a',color='Blue')
p.line(x=ind, y=frame['b'].values, legend_label='b',color='Red')

p.background_fill_color='#dddddd'
p.outline_line_color='white'
p.legend.location = "top_left"
p.legend.click_policy = "hide"

#save(p)
show(p)

from tqdm import tqdm

!pip install psycopg2
import psycopg2
from psycopg2 import Error

start_date= "'" + str(start_date) + "'"
sql='SELECT pv."START_DATE", pp."NAME" from '+str(klient)+'."profile" pv left join '+str(klient)+'."powerp pp on pp."ID" = pv."IDD" left join.... WHERE pv."start_date">=' +start_date+' order by pp."A", pv."B"'
sql = 'SELECT * FROM '+str(name)+'.aaa ORDER BY "ID" ASC '
try:
    connection = psycopg2.connect(host='servername', port='1234', dbname='a', user='A', password='a')
    dat=pd.read_sql_query(sql, connection)
except (Exception, Error) as error:
    print('Błąd połączenia z bazą SQL',error)
finnaly:
    if (connection):
        connection.close()

from datetime import date
date.today().strftime("%Y-%m-%d")

.dt.tz.localize(None)
df['A']=df['A'].apply(lambda x: x.tz_localize('Europe/Warsaw'))
df['A']=df['A'].apply(lambda x: x.tz_convert('UTC'))

.apply(lambda xL pd.to_datetime(df['A'], format='%y-%m-%d %H:%M:%S'))

df['T']=np.where(df['Date']>= dfx['start_date'].iloc[k1], dfx['T'].iloc[k1], df['T'])

sales.groupby("group")["A"].mean()



!pip install xgboost
import xgboost as xgb

y=df5['A'].copy()
X=df5.copy()
X=X.drop(['A'], axis=1)

xg = xgb.XGBRegressor(n_jobs=1,n_estimators=50)
xg.fit(X,y)
prognosis=xg.predict(X)

xgb.plot_importance(xg)



!pip install --upgrade pip setuptools wheel

import shap
import xgboost
from sklearn.model_selection import train_test_split

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu XGBoost
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train, label=y_train), 100)

# Wyjaśnienie modelu przy użyciu SHAP
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)

# Wykres SHAP dla pojedynczej obserwacji
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# Wykres podsumowania SHAP
shap.summary_plot(shap_values, X)
