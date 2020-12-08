import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta


from model import train_model
from database import fetch_all_bpa_as_df

# Definitions of constants. This projects uses extra CSS stylesheet at `./assets/style.css`
COLORS = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/style.css']

# Define the dash app first
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Define component functions


def page_header():
    """
    Returns the page header as a dash `html.Div`
    """
    return html.Div(id='header', children=[
        html.Div([html.H3('Charting The Charts')],
                 className="ten columns"),
        html.A([html.Img(id='logo', src=app.get_asset_url('github.png'),
                         style={'height': '35px', 'paddingTop': '7%'}),
                html.Span('Github Repo', style={'fontSize': '2rem', 'height': '35px', 'bottom': 0,
                                                'paddingLeft': '4px', 'color': '#a3a7b0',
                                                'textDecoration': 'none'})],
               className="two columns row",
               href='https://github.com/ncl11/1050_project'),
    ], className="row")


def description():
    """
    Returns overall project description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
        # About
        ### Project & Executive Summary
        It is no secret that the movie industry has begun to tap into immensely effective techniques through social media and news articles to create conversation online about upcoming movie releases. But how effective is the phenomenon of “online buzz”?

        The recent trend of multi-million blockbuster movies (the annual MCU summer releases come to mind), which were prefaced with non-stop online conversation for months, sometimes years, underlines the idea that a big part of what makes a movie money is how many people can be convinced to watch it. And statistically, if more people are talking about a movie, it will make more people curious about the movie.

        Charting the Charts is a tool to utilize the potential predictive power of online trending data to predict future revenue trends for new movie releases.
        
        The data used in this model is retrieved from two data sources. The BoxOfficeMojo data is scraped and the Google Trends data is available through gtab.  Our database updates automatically every monday with the previous week of data.  Unsurprisingly the most predictive variable for box office gross was the gross from the previous week.  This correlation can be seen below.

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")

#import plotly.graph_objects as go
import pandas as pd

import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta


COLORS = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
c = ['red', 'blue', 'orange', 'white']
def static_stacked_trend_graph(stack=False):
    """
    Returns scatter line plot of all power sources and power load.
    If `stack` is `True`, the 4 power sources are stacked together to show the overall power
    production.
    """

    trends = []
    revenues = []
    
    df = pd.read_csv('3_mo_weekly.csv', sep='\t')
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')
    date = df['Date_dt'].iloc[-1]

    stack = False
    if df is None:
        return go.Figure()
    
    fig = go.Figure()
    movie_list = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)['Release'][0:20]
    for row in range(df.shape[0]):
        date = df['Date_dt'].iloc[-1]

        trend = df.iloc[row]['Weekly']

        rev = df.iloc[row]['Week + 1']
        if rev == 0:
            continue

        trends.append(trend)
        revenues.append(rev)
        fig.add_trace(go.Scatter(x=[trend], y=[rev], mode='markers', name=df.iloc[row]["Release"],
                         line={'width': 2, 'color': c[row%4]},
                         stackgroup='stack' if stack else None))
    trends, revenues = pd.Series(trends), pd.Series(revenues)
    corr = trends.corr(revenues)
    
    title = f'Weekly Gross(week i) vs Weekly Gross(week i+1): Correlation = {corr}'
    if stack:
        title += ' [Stacked]'
    fig.update_layout(template='plotly_dark',
                      title=title,
                      plot_bgcolor='#23272c',
                      paper_bgcolor='#23272c',
                      yaxis_title='Weekly Gross(week i+1)',
                      xaxis_title='Weekly Gross(week i)')
    return fig
#static_stacked_trend_graph()

def description2():
    """
    Returns overall project description in markdown
    """
    return html.Div(children=[dcc.Markdown('''
    The correlation between Google Trend data and gross varies widely from week to week.  The data was highly correlated with the gross for some weeks and not strongly correlated for others.  This is largely dependent on the names of the highest grossing movies and if there is overlap with common search words on Google.  Below are two stacked bar plots of the normalized Gross and Google Trend data for the week beginning 6/22/20 and the most current week respectively.  Some of the issues that arrise when using Google Trend data are discussed below in the 'Next Steps' section. 

        ''', className='eleven columns', style={'paddingLeft': '5%'})], className="row")

def static_stacked_bar_graph(stack=False):
    df = pd.read_csv('6_mo_weekly.csv', sep='\t')
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')

    date = df['Date_dt'].unique()[5]

    fig = go.Figure()
    sorted_df = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)[0:15][::-1]
    y = sorted_df['Release']
    x_rev = sorted_df['Weekly'] / np.sum(sorted_df['Weekly'])
    x_trend = sorted_df['google trends'] / np.sum(sorted_df['google trends'])
    trends, revenues = pd.Series(x_trend), pd.Series(x_rev)
    corr = trends.corr(revenues)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x_rev,
        name='Gross',
        orientation='h',
        marker=dict(
            color='blue', #rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=x_trend,
        name='Google Trends',
        orientation='h',
        marker=dict(
            color='red', #rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    title = f'Normalized Gross and Google Trends for top Movies-Week Starting 6/22/2020: Correlation = {corr}'

    #fig.update_layout(barmode='stack')
    fig.update_layout(template='plotly_dark',
                    title=title,
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    yaxis_title='Movies',
                    xaxis_title='Normalized Quantities',barmode='stack')


    return fig
    
def static_stacked_bar_graph_current(stack=False):
    df = pd.read_csv('3_mo_weekly.csv', sep='\t')
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')

    date = df['Date_dt'].iloc[-1]

    fig = go.Figure()
    sorted_df = df[df['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)[0:15][::-1]
    y = sorted_df['Release']
    x_rev = sorted_df['Weekly'] / np.sum(sorted_df['Weekly'])
    x_trend = sorted_df['google trends'] / np.sum(sorted_df['google trends'])
    trends, revenues = pd.Series(x_trend), pd.Series(x_rev)
    corr = trends.corr(revenues)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x_rev,
        name='Gross',
        orientation='h',
        marker=dict(
            color='blue', #rgba(246, 78, 139, 0.6)',
            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=x_trend,
        name='Google Trends',
        orientation='h',
        marker=dict(
            color='red', #rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    title = f'Normalized Gross and Google Trends for top 15 Movies-Most Recent Data: Correlation = {corr}'

    #fig.update_layout(barmode='stack')
    fig.update_layout(template='plotly_dark',
                    title=title,
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    yaxis_title='Movies',
                    xaxis_title='Normalized Quantities',barmode='stack')


    return fig


def what_if_description():
    """
    Returns description of "What-If" - the interactive component
    """
    return html.Div(children=[
        dcc.Markdown('''
        # Our Model
        Below we have a plot of the top 5 movies from the previous week and our prediction for the upcoming week of gross.  The chart is updated automatically with new gross data and a new prediction every week as new data comes in.  The top slider allows the user to toggle through movies 1 to 5 and the second binary slider shows our prediction.  For this model we have found a simple linear regression with ridge regularization to be very effective.  
        ''', className='eleven columns', style={'paddingLeft': '5%'})
    ], className="row")


def what_if_tool():
    """
    Returns the What-If tool as a dash `html.Div`. The view is a 8:3 division between
    demand-supply plot and rescale sliders.
    """
    return html.Div(children=[
        html.Div(children=[dcc.Graph(id='what-if-figure')], className='nine columns'),

        html.Div(children=[
            html.H5("Current Top 5 Movies", style={'marginTop': '2rem'}),
            html.Div(children=[
                dcc.Slider(id='wind-scale-slider', min=0, max=4, step=1, value=0, className='row',
                           marks={x: str(x) for x in np.arange(0, 4.1, 1)})
            ], style={'marginTop': '5rem'}),

            html.Div(id='wind-scale-text', style={'marginTop': '1rem'}),

            html.Div(children=[
                dcc.Slider(id='hydro-scale-slider', min=0, max=1, step=1, value=0,
                           className='row', marks={x: str(x) for x in np.arange(0, 1.1, 1)})
            ], style={'marginTop': '3rem'}),
            html.Div(id='hydro-scale-text', style={'marginTop': '1rem'}),
        ], className='three columns', style={'marginLeft': 5, 'marginTop': '10%'}),
    ], className='row eleven columns')


def architecture_summary():
    """
    Returns the text and image of architecture summary of the project.
    """
    return html.Div(children=[
        dcc.Markdown('''
            # Possible next steps

            In terms of next steps, we would hope to mitigate some of the biases that arise when using “Google Trends” as a predictor. Given the complexity of some movie titles, it is unlikely that people will search for these movies word for word. Instead, we would want to account for possible abbreviations and related searches for these movies. Conversely, some movie titles are also commonly used words/phrases. In these situations the Google Trend data underestimates and overestimates, respectively, the number of searches for the movie. If given more time, we would want to build a more robust model that lessens the effects of the aforementioned issues. 

            Ridge regression was very effective for our data but in the future we may look to more sophisticated hyperparameter tuning to see if we can improve the predictive power using other machine learning models.  To improve the model approach, it would also be a good idea to incorporate more data besides just using previous week data to predict the next week. By adding in a previous two weeks of data we may have been able to achieve higher accuracy. 

            # Additional Information

            ### Datasets Used 
            We acquired data from the following sites:

            * https://trends.google.com/trends/?geo=US 
            * https://www.boxofficemojo.com/?ref_=bo_nb_da_mojologo
            
            We obtained daily searches from Google Trends and daily gross from BoxOfficeMojo and merged the two data sets on date and movie title.  Our dataset will be automatically updated at weekly intervals through rescraping BoxOfficeMojo and merging with the new Google Trends data. 

            ### Development Process and Final Technology Stack

            Our website was created and hosted through Gitpod. Given that Gitpod is powered by VS Code, the site was created accordingly. 

            ### Data Acquisition, Caching, ETL Processing, Database Design

            As mentioned above, we obtained daily searches from Google Trends and daily gross from BoxOfficeMojo and merged the two data sets on date and movie title. Please find our ETL steps below.

            Extract - Data was extracted by scraping BoxOfficeMojo as well as Google Trends. 

            Transform - Once the data was scraped, the resulting data frames were merged so that one data frame contained revenue and trend information for a specific movie. Data was cleansed, stripped of extraneous characters, and converted into workable types. 

            Load/Descriptor of Database Used -  Given the relatively small CSV files in our problem, we have elected not to host it on MongoDB. However, if the data files in question were larger, we would hope to use MongoDB to establish a database.  Furthermore, given the predictive model we ran, we found it easier to read CSV files locally into a pandas data frame. 

            Link to a static version of your ETL_EDA.ipynb notebook, or equivalent web page
            https://github.com/ncl11/1050_project/blob/master/ETL_EDA.ipynb
            https://github.com/ncl11/1050_project/blob/master/ETL_EDA2.ipynb

            Link to a static version of your Enhancement.ipynb notebook, or equivalent web page
            https://github.com/ncl11/1050_project/blob/master/Enhancement.ipynb



        ''', className='row eleven columns', style={'paddingLeft': '5%'}),


        dcc.Markdown('''
        
        ''')
    ], className='row')


# Sequentially add page components to the app's layout
def dynamic_layout():
    return html.Div([
        page_header(),
        html.Hr(),
        description(),
        # dcc.Graph(id='trend-graph', figure=static_stacked_trend_graph(stack=False)),
        dcc.Graph(id='stacked-trend-graph', figure=static_stacked_trend_graph(stack=True)),
        description2(), 
        dcc.Graph(id='stacked-trend-graph2', figure=static_stacked_bar_graph(stack=False)),
        dcc.Graph(id='stacked-trend-graph3', figure=static_stacked_bar_graph_current(stack=False)),
        what_if_description(),
        what_if_tool(),
        architecture_summary(),
    ], className='row', id='content')


# set layout to a function which updates upon reloading
app.layout = dynamic_layout


# Defines the dependencies of interactive components

@app.callback(
    dash.dependencies.Output('wind-scale-text', 'children'),
    [dash.dependencies.Input('wind-scale-slider', 'value')])
def update_wind_sacle_text(value):
    """Changes the display text of the wind slider"""
    return "Movie {:.2f}".format(value)


@app.callback(
    dash.dependencies.Output('hydro-scale-text', 'children'),
    [dash.dependencies.Input('hydro-scale-slider', 'value')])
def update_hydro_sacle_text(value):
    """Changes the display text of the hydro slider"""
    return "Show Prediction: On/Off {:.2f}".format(value)



@app.callback(
    dash.dependencies.Output('what-if-figure', 'figure'),
    [dash.dependencies.Input('wind-scale-slider', 'value'),
     dash.dependencies.Input('hydro-scale-slider', 'value')])
def what_if_handler(wind, hydro):
    stack = False
    df = pd.read_csv('3_mo_weekly.csv', sep='\t')
    df['Date_dt'] = df['Date'].astype('datetime64[ns]')
    date = df['Date_dt'].iloc[-1]

    df_test = df.groupby('Release').filter(lambda x : x['Release'].shape[0]>=2)
    movie_list = df_test[df_test['Date_dt'] == date].sort_values(by=['Weekly'], ascending=False)['Release']

    wind = int(wind)
    hydro = int(hydro)
    df = df[df['Release'] == movie_list.iloc[wind]]
    if df is None:
        return go.Figure()
    sources = ['Wind', 'Hydro', 'Fossil/Biomass', 'Nuclear']
    x = df['Date']

    predict_date = df['Date_dt'].iloc[-1]  + timedelta(days=7)
    y_pred = train_model().predict([df.drop(['Release', 'Date', 'Y', 'Week + 1', 'Date_dt'], axis=1).iloc[-1]])

    fig = go.Figure()
    #for i, s in enumerate(sources):
    fig.add_trace(go.Scatter(x=x, y=df['Weekly'], mode='lines', name=movie_list.iloc[wind],
                             line={'width': 2, 'color': 'orange'},
                             stackgroup='stack' if stack else None))
    if hydro:
        fig.add_trace(go.Scatter(x=[predict_date], y=y_pred, mode='markers', name='Prediction',
                                line={'width': 2, 'color': 'red'},
                                stackgroup='stack' if stack else None))
                                
    #fig.update_layout(yaxis=dict(range=[0, 1.2*df['Weekly'].max()]))
    fig.update_layout(yaxis=dict(range=[0, 1.2*df['Weekly'].max()]), xaxis=dict(range=[x.iloc[0], predict_date+timedelta(days=1)]))


    title = f'Weekly Revenue for {movie_list.iloc[wind]}'
    if stack:
        title += ' [Stacked]'
    fig.update_layout(template='plotly_dark',
                      title=title,
                      plot_bgcolor='#23272c',
                      paper_bgcolor='#23272c',
                      yaxis_title='MW',
                      xaxis_title='Date/Time')


 #   """Changes the display graph of supply-demand"""
 #   df = fetch_all_bpa_as_df(allow_cached=True)
 #   x = df['Datetime']
 #   supply = df['Wind'] * wind + df['Hydro'] * hydro + df['Fossil/Biomass'] + df['Nuclear']
 #   load = df['Load']

 #   fig = go.Figure()
 #   fig.add_trace(go.Scatter(x=x, y=supply, mode='none', name='supply', line={'width': 2, 'color': 'pink'},
 #                 fill='tozeroy'))
 #   fig.add_trace(go.Scatter(x=x, y=load, mode='none', name='demand', line={'width': 2, 'color': 'orange'},
 #                 fill='tonexty'))
 #   fig.update_layout(template='plotly_dark', title='Supply/Demand after Power Scaling',
 #                     plot_bgcolor='#23272c', paper_bgcolor='#23272c', yaxis_title='MW',
 #                     xaxis_title='Date/Time')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=1050, host='0.0.0.0')
