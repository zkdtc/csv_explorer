
import math
import numpy as np
import pandas as pd
from io import BytesIO
import os
import base64
from math import pi

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, cumsum
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.palettes import Category20c, Turbo256, RdYlBu

#import holoviews as hv
#hv.extension('bokeh')
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics

df_new = pd.read_csv('../data/dataset_1911.csv')
df = pd.read_csv('../data/titanic.csv')
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])

##  Function to calculate r2_score and RMSE on train and test data
def get_model_score(model, X_train, y_train, X_test, y_test, flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[] 
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_r2=metrics.r2_score(y_train,pred_train)
    test_r2=metrics.r2_score(y_test,pred_test)
    train_rmse=np.sqrt(metrics.mean_squared_error(y_train,pred_train))
    test_rmse=np.sqrt(metrics.mean_squared_error(y_test,pred_test))
    
    #Adding all scores in the list
    score_list.extend((train_r2,test_r2,train_rmse,test_rmse))
    
    # If the flag is set to True then only the following print statements will be dispayed, the default value is True
    if flag==True: 
        print("R-sqaure on training set : ",metrics.r2_score(y_train,pred_train))
        print("R-square on test set : ",metrics.r2_score(y_test,pred_test))
        print("RMSE on training set : ",np.sqrt(metrics.mean_squared_error(y_train,pred_train)))
        print("RMSE on test set : ",np.sqrt(metrics.mean_squared_error(y_test,pred_test)))
    
    # returning the list with train and test scores
    return score_list


palette = ['#ba32a0', '#f85479', '#f8c260', '#00c2ba']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'


def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None

def heatmap_chart_plot(dataset, cpalette = RdYlBu[11]):

    # Feature engineering
    #df_feature = dataset.drop(['Adjusted_demand', 'Sales_depot_ID', 'Sales_channel_ID'], axis = 1)
    #df_feature['Sales_unit'] = df_feature['Sales_unit'].astype(float)
    #df_feature['price'] = df_feature['Sales']/df_feature['Sales_unit']
    ## bimbo_1911['Demand_revenue'] = bimbo_1911['price']*bimbo_1911['Demanda_uni_equil']
    #df_feature.price.fillna(0, inplace = True)
    #df_corr = df_feature.corr()
    df_corr = session['df'].corr()
    df = pd.DataFrame(df_corr.stack(), columns=['p']).reset_index()
    #import pdb;pdb.set_trace()
    x_labels = list(df_corr.index)
    y_labels = list(df_corr.columns)
    colors = cpalette # ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=cpalette, low=-0.5, high=1)

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    p = figure(title="Features Correlation Heatmap",
               x_range=x_labels, y_range=y_labels,
               x_axis_location="above", plot_width=600, plot_height=600,
               tools=TOOLS, toolbar_location='below',
               tooltips=[('Feature1', '@level_0'), ('Feature2', ' @level_1'), ('p', '@p')])
    
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "14px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="level_0", y="level_1", width=1, height=1,
            source=df,
            fill_color={'field': 'p', 'transform': mapper},
            line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="14px",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%.1f"),
                     label_standoff=6, border_line_color=None)
    p.add_layout(color_bar, 'right')
    return p
    
def pie_chart_plot(dataset,cpalette=palette):
    df_feature = dataset.drop(['Adjusted_demand', 'Sales_depot_ID', 'Sales_channel_ID'], axis = 1)
    df_feature['Sales_unit'] = df_feature['Sales_unit'].astype(float)
    df_feature['price'] = df_feature['Sales']/df_feature['Sales_unit']
    # bimbo_1911['Demand_revenue'] = bimbo_1911['price']*bimbo_1911['Demanda_uni_equil']
    df_feature.price.fillna(0, inplace = True)

    data = df_feature.groupby('Product_name').size()# this is a Series #pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
    data = data.reset_index(name='value').rename(columns={'index':'Product_name'}) # Change a Series to a df
    data = data.sort_values(by=['value'], ascending=False)
    # Truncate the top 20
    data = data.head(20)
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = Category20c[20][0:len(data)]
    #import pdb;pdb.set_trace()
    p = figure(plot_height=350, title="Sales Pie Chart", toolbar_location=None,
              tools="hover", tooltips="@Product_name: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='Product_name', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    return p

def histogram_general(cpalette=palette):
    df = session['df']
    if 'columnSelectedHist' in request.form:
        column_selected = request.form['columnSelectedHist'] 
        measured = df[column_selected]
    else:
        column_selected = df.columns[0]
        measured = df[column_selected]
    
    hist, edges = np.histogram(measured, density=True, bins=50)
    p = figure(title=column_selected, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    #p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
    #p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    return p
    return p

def redraw(pieColumn):
    heatmap_chart = heatmap_chart_plot(df_new)
    pie_chart = pie_chart_plot(df_new)
    hist_chart = histogram_general()
    return (
        heatmap_chart,
        pie_chart,
        hist_chart
    )


app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

UPLOAD_FOLDER = '../data/'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def Home():
    

    return render_template(
        'index.html'
    )

#@app.route('/upload')
#def upload_file():
#   return render_template('upload.html')
	
#@app.route('/uploader', methods = ['GET', 'POST'])
#def upload_file():
#   if request.method == 'POST':
#      f = request.files['file']
#      f.save(secure_filename(f.filename))
#      return 'file uploaded successfully'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file_list=[]
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for filename in files:
            file_list.append(filename)

    if request.method == 'POST' and 'fileSelected' in request.form:
        # Load the file
        df = pd.read_csv(UPLOAD_FOLDER+'/'+request.form['fileSelected'])
        session['df']=df
        flash('Successfully loaded '+request.form['fileSelected'])

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    return render_template('upload.html',file_list = file_list)


@app.route('/eda', methods=['GET', 'POST'])
def EDA():
    options_list=df_new.columns
    options_list_hist = session['df'].columns

    heatmap_chart,pie_chart,hist_chart = redraw(1)

    script_heatmap_chart, div_heatmap_chart = components(heatmap_chart)
    script_pie_chart, div_pie_chart = components(pie_chart)
    script_hist_chart, div_hist_chart = components(hist_chart)

    return render_template(
        'eda.html',
        div_heatmap_chart=div_heatmap_chart,
        script_heatmap_chart=script_heatmap_chart, 
        div_pie_chart=div_pie_chart,
        script_pie_chart=script_pie_chart,
        div_hist_chart=div_hist_chart,
        script_hist_chart=script_hist_chart,
        options_list=options_list,
        options_list_hist=options_list_hist
    )

@app.route('/model', methods=['GET', 'POST'])
def ModelData():
    features_list = session['df'].columns
    if 'mymultiselect' in request.form:
        multiselect = request.form.getlist('mymultiselect')
        session['FeaturesForModel'] = multiselect
        s = ''
        for i in range(len(multiselect)):
            s+=multiselect[i]+' '
        flash('Successfully set '+s+'as input features')

        targetselect = request.form.getlist('targetselect')
        session['TargetForModel'] = targetselect
        s2 = ''
        for i in range(len(targetselect)):
            s2+=targetselect[i]+' '
        flash('Successfully set '+s2+'as target')   

        flash('Begin Training')
        df = session['df']
        X = df[multiselect]
        y = df[targetselect]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)
        rf_estimator=RandomForestRegressor(random_state=1)
        rf_estimator.fit(X_train,y_train)
        rf_estimator_score=get_model_score(rf_estimator, X_train, y_train, X_test, y_test)
        s3=''
        labels = ['train r2=','test r2=','train_rmse=','test_rmse=']
        for i in range(len(rf_estimator_score)):
            template = labels[i]+'{num:.2f}'
            s3+=template.format(num=rf_estimator_score[i])+' '
        flash('Model Scores:'+s3)
        session['model']=rf_estimator

    return render_template(
        'model.html',
        features_list = features_list
    )

@app.route('/predict', methods=['GET', 'POST'])
def Predict():
    show_message = ''
    features_list = session['FeaturesForModel']
    model = session['model']
    if 'message' in session:
        show_message = session['message']

    if features_list[0]+'/' in request.form:
        x=[]
        s_features = ''
        for feature in features_list:
            value_this = str(request.form[feature+'/'])+'\n'
            s_features += feature+'='+value_this
            x.append(value_this)

        y = model.predict([x])

        session['message'] = 'With '+s_features+', \nPredicted '+session['TargetForModel'][0]+'='+str(y[0])


    return render_template(
        'predict.html', features_list = features_list, show_message = show_message
    )
if __name__ == '__main__':
    app.run(debug=True)


