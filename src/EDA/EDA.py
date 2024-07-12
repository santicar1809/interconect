import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    describe_result=data.describe()
    
    eda_path = './files/modeling_output/figures/'

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # Exporting the file
    with open(eda_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open(eda_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
    
    # ¿Cuales son los medios de pago más utilizados?
    df_payment_method=data.groupby(['payment_method','senior_citizen'],as_index=False)['customer_id'].count()
    df_payment_method.sort_values(by='customer_id',ascending=False,inplace=True)
    fig1=px.histogram(df_payment_method,x='payment_method',y='customer_id',color='senior_citizen',barmode='group')
    fig1.write_image(eda_path+'fig1.png', format='png', scale=2)
    fig1.write_html(eda_path+'fig1.html') 
    
    # ¿Que tipo de factura prefieren los usuarios?
    df_paperless_billing=data.groupby(['paperless_billing','senior_citizen'],as_index=False)['customer_id'].count()
    df_paperless_billing.sort_values(by='customer_id',ascending=False,inplace=True)
    fig2=px.histogram(df_paperless_billing,x='paperless_billing',y='customer_id',color='senior_citizen',barmode='group')
    fig2.write_image(eda_path+'fig2.png', format='png', scale=2)
    fig2.write_html(eda_path+'fig2.html')
    
    # ¿Que tipo de contrato prefieren las personas que tienen personas a cargo?
    df_total_charges=data.groupby(['type','dependents'],as_index=False)['total_charges'].sum()
    df_total_charges.sort_values(by='total_charges',ascending=False,inplace=True)
    fig3=px.histogram(df_total_charges,y='type',x='total_charges',color='dependents',barmode='group')
    fig3.write_image(eda_path+'fig3.png', format='png', scale=2)
    fig3.write_html(eda_path+'fig3.html')
    
    df_internet_service_1=data.groupby(['internet_service','online_security'],as_index=False)['total_charges'].sum()
    df_internet_service_1.sort_values(by='total_charges',ascending=False,inplace=True)
    fig4=px.histogram(df_internet_service_1,y='internet_service',x='total_charges',color='online_security',barmode='group')
    fig4.write_image(eda_path+'fig4.png', format='png', scale=2)
    fig4.write_html(eda_path+'fig4.html')
    
    df_internet_service_1=data.groupby(['internet_service','online_security'],as_index=False)['customer_id'].count()
    df_internet_service_1.sort_values(by='customer_id',ascending=False,inplace=True)
    fig5=px.histogram(df_internet_service_1,y='internet_service',x='customer_id',color='online_security',barmode='group')
    fig5.write_image(eda_path+'fig5.png', format='png', scale=2)
    fig5.write_html(eda_path+'fig5.html')
    
    ## ¿Cual es la tasa de cancelación y que caracteristicas tienen los usuarios que cancelan y los que no?
    
    data_cancel_rate=data.groupby(['cancel'])['customer_id'].count()
    data_cancel_rate.plot(kind='bar')
    plt.show()
    
    df_internet_service_2=data.groupby(['internet_service','cancel'],as_index=False)['customer_id'].count()
    df_internet_service_2.sort_values(by='cancel',ascending=False,inplace=True)
    fig6=px.histogram(df_internet_service_2,y='internet_service',x='customer_id',color='cancel',barmode='group')
    fig6.write_image(eda_path+'fig6.png', format='png', scale=2)
    fig6.write_html(eda_path+'fig6.html')
    
    df_payment_type=data.groupby(['type','cancel'],as_index=False)['customer_id'].count()
    df_payment_type.sort_values(by='cancel',ascending=False,inplace=True)
    fig7=px.histogram(df_payment_type,y='type',x='customer_id',color='cancel',barmode='group')
    fig7.write_image(eda_path+'fig7.png', format='png', scale=2)
    fig7.write_html(eda_path+'fig7.html')
    
    df_expenses=data.groupby(['type','cancel'],as_index=False)['total_charges'].sum()
    df_expenses.sort_values(by='cancel',ascending=False,inplace=True)
    fig8=px.histogram(df_expenses,y='type',x='total_charges',color='cancel',barmode='group')
    fig8.write_image(eda_path+'fig8.png', format='png', scale=2)
    fig8.write_html(eda_path+'fig8.html')
    
    df_multiline=data.groupby(['cancel','multiple_lines'],as_index=False)['customer_id'].count()
    df_multiline.sort_values(by='cancel',ascending=False,inplace=True)
    fig9=px.histogram(df_multiline,y='cancel',x='customer_id',color='multiple_lines',barmode='group')
    fig9.write_image(eda_path+'fig9.png', format='png', scale=2)
    fig9.write_html(eda_path+'fig9.html')
    
    df_p_method=data.groupby(['payment_method','cancel'],as_index=False)['customer_id'].count()
    df_p_method.sort_values(by='cancel',ascending=False,inplace=True)
    fig10=px.histogram(df_p_method,y='payment_method',x='customer_id',color='cancel',barmode='group')
    fig10.write_image(eda_path+'fig10.png', format='png', scale=2)
    fig10.write_html(eda_path+'fig10.html')
    
    df_multiline=data.groupby(['cancel','multiple_lines'],as_index=False)['customer_id'].count()
    df_multiline.sort_values(by='cancel',ascending=False,inplace=True)
    fig11=px.histogram(df_multiline,y='cancel',x='customer_id',color='multiple_lines',barmode='group')
    fig11.write_image(eda_path+'fig11.png', format='png', scale=2)
    fig11.write_html(eda_path+'fig11.html')