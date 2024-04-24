import dash
from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
from scipy.interpolate import griddata
from sklearn.neural_network import MLPRegressor

import base64

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "1rem 1rem",
    "font-size": "1.1em",
    "background-color": "#333333",
}

CONTENT_STYLE = {
    "margin-left": "8%",
    "margin-right": "8%",
    "font-size": "1.1em"
}

def generate_table(dataframe):
    return html.Div([dash_table.DataTable(dataframe.to_dict('records'), [{"name": i, "id": i} for i in dataframe.columns],
        filter_action="native", id='datatable-interactivity', sort_action="native",
        row_selectable="multi", selected_rows=[],
        sort_mode="multi", page_size=12)],
        style={'maxWidth': '95%', 'maxHeight': '700px', "overflow": "scroll"})

class StaticDashboard():
    def __init__(self, data_path, mode='Release'):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self.app.config['suppress_callback_exceptions'] = True
        self.mode = mode
        self.data_path = data_path
        with open(self.data_path) as json_data:
            self.data = json.load(json_data)

        self.init = False
        self.task_name = None
        self.dfSDI = None
        self.best_value = None
        self.float_parameters_names = None
        self.trial_number = None
        self.float_parameters_bounds = None
        self.discrete_parameters_names = None
        self.discrete_parameters_valid_values = None
        self.best_trial_number = None
        self.trial_names = None
        self.values_update_best = None
        self.points_update_best = None
        self.importance = None
        self.sidebar = None
        self.content = None
        self.global_iter_number = None
        self.local_iter_number = None
        self.accuracy = None
        self.solve_time = None
        self.best_float_point_dictionary = None
        self.best_discrete_point_dictionary = None
        self.optimization_time = None
        self.value_colors_dictionary = None
        self.values_names_T = None
        self.parameter_eps = None
        self.parameter_r = None
        self.parameter_iters_limit = None
        self.values_T_dictionary = None

        self.obj_func_norm = None

        self.functional_count = 1

        self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")]
        )(self.render_page_content)

        self.app.callback(
            Output('crossfilter-indicator-scatter-2', 'figure'),
            Input('crossfilter-xaxis-column-2', 'value')
        )(self.update_graph_1)

        self.app.callback(
            Output('crossfilter-indicator-scatter', 'figure'),
            Input('crossfilter-xaxis-column', 'value')
        )(self.update_graph_2)

        self.app.callback(
            Output('crossfilter-indicator-scatter-1', 'figure'),
            Input('crossfilter-xaxis-column-1', 'value'),
            Input('crossfilter-yaxis-column-1', 'value'),
            Input('crossfilter-type-1', 'value'),
            Input('crossfilter-type-2', 'value'),
        )(self.update_graph_3)

        self.app.callback(
            Output('crossfilter-indicator-scatter-3', 'figure'),
            Input('crossfilter-xaxis-column-3', 'value')
        )(self.update_graph_4)

        self.app.callback(
            Output('output-data-upload', 'children'),
            Input('upload-data', 'contents')
        )(self.update_output)

        self.app.callback(
            Output('datatable-interactivity-container', "children"),
            Input('datatable-interactivity', "derived_virtual_data"),
            Input('datatable-interactivity', "derived_virtual_selected_rows"))(self.update_arhive_graph)

    def readData(self, data):
        self.parameter_eps = data['Parameters'][0]['eps']
        self.parameter_r = data['Parameters'][0]['r']
        self.parameter_iters_limit = data['Parameters'][0]['iters_limit']
        self.task_name = data['Task'][0]['name']

        self.functional_count = len(data['SearchDataItem'][0]["function_values"])

        self.dfSDI = pd.DataFrame(data['SearchDataItem'])
        dfFV = pd.DataFrame(data['Task'][0]['float_variables'])
        dfDV = pd.DataFrame(data['Task'][0]['discrete_variables'])
        dfBT = pd.DataFrame(data['best_trials'])
        dfS = pd.DataFrame(data['solution'])

        self.float_parameters_bounds = data['Task'][0]['float_variables']
        self.discrete_parameters_valid_values = data['Task'][0]['discrete_variables']

        self.float_parameters_names = [f"{x} [fv]" for x in [*dfFV]]
        self.discrete_parameters_names = [f"{x} [dv]" for x in [*dfDV]]

        self.parameters_names = self.float_parameters_names + self.discrete_parameters_names

        self.best_value = dfBT['__z'][0]
        self.accuracy = float(dfS['solution_accuracy'])
        self.trial_number = int(dfS['number_of_trials'])
        self.global_iter_number = int(dfS['number_of_global_trials'])
        self.local_iter_number = int(dfS['number_of_local_trials'])
        self.solve_time = float(dfS['solving_time'])
        self.best_trial_number = int(dfS['num_iteration_best_trial'][0][0])

        FVs = pd.DataFrame(self.dfSDI['float_variables'].to_list(), columns=self.float_parameters_names,
                           index=self.dfSDI.index)
        DVs = pd.DataFrame(self.dfSDI['discrete_variables'].to_list(), columns=self.discrete_parameters_names,
                           index=self.dfSDI.index)

        self.dfSDI = self.dfSDI[["__z", "creation_time", "x", "delta", "globalR", "localR"]]
        self.dfSDI = pd.concat([DVs, self.dfSDI], axis=1, join='inner')
        self.dfSDI = pd.concat([FVs, self.dfSDI], axis=1, join='inner')

        boundary_points_indexes = self.dfSDI[self.dfSDI['__z'] >= 1.797692e+308].index
        self.dfSDI.drop(boundary_points_indexes, inplace=True)
        self.dfSDI.insert(loc=0, column='trial', value=np.arange(1, len(self.dfSDI) + 1))

        accuracy_in_signs = 0
        accuracy_copy = self.accuracy
        while accuracy_copy < 1:
            accuracy_copy *= 10
            accuracy_in_signs += 1

        self.best_float_point_dictionary = dict(
            zip(self.float_parameters_names, [round(elem, accuracy_in_signs) for elem in dfBT['float_variables'][0]]))
        self.best_discrete_point_dictionary = dict(zip(self.discrete_parameters_names, dfBT['discrete_variables'][0]))

        self.optimization_time = self.dfSDI["creation_time"].to_list()

        isFirst = 1
        shift = 0
        new_times = []
        for elem in self.optimization_time:
            if elem != 0 and isFirst:
                shift = elem
                isFirst = 0
            if elem != 0:
                elem -= shift
            new_times.append(elem)

        self.optimization_time = new_times
        self.trial_names = ['trial ' + str(index + 1) for index in range(self.trial_number)]

    def calculateLowerEnvelopeValues(self):
        self.values_update_best = []
        self.points_update_best = []
        min_value = max(self.dfSDI['__z']) + 1
        for value, point in zip(self.dfSDI['__z'], self.dfSDI['trial']):
            if min_value > value:
                min_value = value
                self.values_update_best.append(value)
                self.points_update_best.append(point)
        self.values_update_best.append(self.best_value)
        self.points_update_best.append(self.trial_number)


    def calculateParametersImportance(self):
        minZ = min(self.dfSDI['__z'])
        maxZ = max(self.dfSDI['__z'])

        self.importance = []
        for parameter_name in self.parameters_names:
            uniq_parameter_values = self.dfSDI[parameter_name].unique()
            parameter_importance_value = 0
            for value in uniq_parameter_values:
                data = self.dfSDI.loc[self.dfSDI[parameter_name] == value]
                parameter_importance_value += (max(data['__z']) - min(data['__z']))
            self.importance.append(round(parameter_importance_value / (len(uniq_parameter_values) * (maxZ - minZ)), 2))

    def createSidebar(self):
        self.sidebar = html.Div(
            [
                dbc.Navbar(dbc.Container([dbc.Col([dbc.Nav([
                        dbc.NavItem(html.Img(src=r'assets/iOptdash_light_.png', alt='image', height="50px")),
                        dbc.NavItem(dbc.NavLink("Solution", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Analytics", href="/page-1", active="exact")),
                        dbc.NavItem(
                            dbc.NavLink("Archive", href="/page-2", active="exact"),
                            # add an auto margin after page 2 to
                            # push later links to end of nav
                            className="me-auto",
                        ),
                        dbc.NavItem(dbc.NavLink("About", href="https://iopt.readthedocs.io/ru/latest")),
                        dbc.NavItem(dbc.NavLink("Github Source", href="https://github.com/aimclub/iOpt"))

                    ])])]),
                    sticky="bottom",
                    color="primary",
                    dark=True,
                ),
            ], className="sidebar discription"
        )

    def createContentPage(self):
        self.content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

    def launch(self):
        self.readData(self.data)
        del self.data
        self.data = None
        self.calculateLowerEnvelopeValues()
        self.calculateParametersImportance()

        self.createSidebar()
        self.createContentPage()
        self.app.layout = html.Div([
            dcc.Location(id="url"),
            self.sidebar,
            self.content
        ], style={"background-color": "#FBFBFB"})
        #url = "http://127.0.0.1:8050/"
        self.app.run(debug=True)

    def render_task_discription(self):
        return html.Div(children=[
            html.H2('PROBLEM DISCRIPTION',
                    style={'textAlign': 'left'}),
            html.P(f"Problem name: {self.task_name}", style={'color': '#212121'}),
            html.P(f"Functionals count: objective = 1, constraint = {self.functional_count - 1}", style={'color': '#212121'}),
            html.P(f"Parameters count: float = {len(self.float_parameters_names)}, discrete (categorical) = {len(self.discrete_parameters_names)}", style={'color': '#212121'}),
            dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '95%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
            ),
            html.Div(id='output-data-upload')
            ],
            style={'width': '55%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", }
        )
    def render_parameters_discription(self):
        return html.Div(children=[
            html.H2('PARAMETERS USED',
                    style={'textAlign': 'left'}),
            html.P(f"Required accuracy: eps = {self.parameter_eps}",
                   style={'color': '#212121'}),
            html.P(f"Reliability parameter: r = {self.parameter_r}",
                   style={'color': '#212121'}),
            html.P(f"Limit on number of iterations: iters_limit = {self.parameter_iters_limit}",
                   style={'color': '#212121'}),
            ],
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", }
        )
    # "border": "3px #FFFFFF outset"
    def render_solution_discription(self):
        return html.Div(children=[
            html.H2('Solution',
                    style={'textAlign': 'left'}),
            html.P(
                f"Total trial number: {self.trial_number} (global trial number - {self.global_iter_number}, local trial number - {self.local_iter_number})",
                style={'color': '#212121'}),
            html.P(f"{round(self.best_value, 6)}", style={'font-size': '2.0em', 'color': 'black'}),
            html.P(f"Best point: {self.best_float_point_dictionary}, {self.best_discrete_point_dictionary}",
                   style={'color': '#212121'}),
            html.P(f'Best trial number: {self.best_trial_number}', style={'color': '#212121'}),
            html.P(f"Accuracy: {round(self.accuracy, 6)}", style={'color': '#212121'}),
            html.P(f"*[fv] - float variables, [dv] - discrete variables",
                   style={'color': '#212121', 'font-size': '0.8em'}),
        ], className="best trial discription",
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def render_optimization_time_plot(self):
        return html.Div(children=[
            html.H2('Optimization Time',
                    style={'textAlign': 'left'}),

            html.P(f"Total optimization time: {round(self.solve_time, 3)} sec.", style={'color': '#212121'}),

            html.Div(children=[
                dcc.Graph(
                    figure={
                        "data": [{
                            "x": self.dfSDI['trial'],
                            "y": self.optimization_time,
                            'type': 'lines',
                            'marker': {'color': '#3E59A5'}
                        }, ],
                        "layout": {
                            'paper_bgcolor': '#FFFFFF',
                            'plot_bgcolor': '#FFFFFF',
                            'xaxis': {'anchor': 'y', 'title': {'text': 'trial number'}},
                            'yaxis': {'anchor': 'x', 'title': {'text': 'time before calculate trial, sec.'}}
                        },
                    },
                    config={
                        'displayModeBar': True,  # True, False, 'hover'
                    },
                )
            ])
        ], style={'width': '55%', "background-color": "#FFFFFF",
                  "border": "20px solid #FFFFFF", })  # "border":"1px  #FFFFFF outset",

    def render_iteration_characteristic(self):
        return html.Div([
            html.H2('Objective Function Value Сhange',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure=((px.scatter(self.dfSDI, x='trial', y='__z', color_discrete_sequence=['#3E59A5'], marginal_y="histogram", trendline="expanding", trendline_options=dict(function="min"))).update_layout(legend= {'orientation': "h", 'y': -0.25},
                                xaxis={'anchor': 'x', 'title': {'text': 'trial number'}},
                                yaxis={'anchor': 'y', 'title': {'text': 'objective function'}},
                                paper_bgcolor='#FFFFFF',
                                plot_bgcolor='#FFFFFF')),
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def getScatterMatrix(self):
        fig = px.scatter_matrix(self.dfSDI, dimensions=self.parameters_names, color="__z", opacity=0.7,
                                color_continuous_scale="Ice", width=len(self.parameters_names) * 200,
                                height=len(self.parameters_names) * 200)
        fig.update_traces(diagonal_visible=False)
        '''
        fig = ff.create_scatterplotmatrix(self.dfSDI[self.parameters_names+['__z']], diag='histogram', index='__z',
                                          opacity=0.9, colormap=px.colors.sequential.ice, width=len(self.parameters_names) * 200,
                                          height=len(self.parameters_names) * 200)
        '''

        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        return fig

    def render_dependence(self):
        return html.Div([
            html.H2('Parameters Dependence',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure=self.getScatterMatrix(),
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '90%'}),
            ], style={'maxWidth': '90%', 'maxHeight': '650px', "overflow": "scroll"}),
        ], style={'width': '65%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", 'height': '750px'})

    def render_importance(self):
        return html.Div([
            html.H2('Parameters Importance',
                    style={'textAlign': 'left'}),
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': self.importance,
                             'y': self.parameters_names,
                             'orientation': 'h',
                             'type': 'bar',
                             'text': self.importance,
                             'textposition': 'inside',
                             'textfont' : "black",
                             'text_auto':'.2s',
                             'marker': {
                                 'color': '#3E59A5',
                                 'line': {'color': '#3E59A5', 'width': '1'}
                             }
                             },
                        ],
                        'layout': {
                            'xaxis': {
                                'range': [0, 1],
                                'anchor': 'y',
                                'title': {'text': 'trial number'},
                                'tickfont': {'size': '10'},
                            },
                            'yaxis': {
                                'anchor': 'x',
                                'title': {'text': 'parameter'},
                                'tickfont': {'size': '10'}
                            },
                            'title': "Importance of the hyperparameter",
                            'paper_bgcolor': '#FFFFFF',
                            'plot_bgcolor': '#FFFFFF',
                        }
                    },
                    config={
                        'scrollZoom': False,  # True, False
                        'showTips': False,  # True, False
                        'displayModeBar': False,  # True, False, 'hover'
                    },
                )
            ])
        ], style={'width':'35%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF"})

    def render_multidimentional_visualization(self):
        return html.Div(children=[
            html.H2('Multidimentional Visualization',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    html.Br(),
                    html.Label('Parameters For Visualization'),
                    dcc.Dropdown(
                        self.parameters_names,
                        self.parameters_names,
                        id='crossfilter-xaxis-column-3',
                        multi=True
                    ),
                ], style={'width': '15%'}),
                html.Div(children=[
                    dcc.Graph(
                        id='crossfilter-indicator-scatter-3',
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '85%'}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'height': '90%'})
        ], style={ "background-color": "#FFFFFF", "border": "20px solid #FFFFFF"})

    def render_3D_graph(self):
        return html.Div(children=[
            html.H2('Visualization in cross-section of the best discrete parameters',
                    style={'textAlign': 'left'}),
            html.Div([
                html.Div(children=[
                    html.Br(),
                    html.Label('Type'),
                    dcc.Dropdown(
                        ['3D Surface', 'Heatmap'],
                        '3D Surface',
                        id='crossfilter-type-1',
                    ),
                    html.Label('Calculate mode'),
                    dcc.Dropdown(
                        ['none (by trials points)', 'interpolation', 'approximation'],
                        'interpolation',
                        id='crossfilter-type-2',
                    ),
                    html.Label('X-axis parameter'),
                    dcc.Dropdown(
                        self.float_parameters_names,
                        self.float_parameters_names[0],
                        id='crossfilter-xaxis-column-1',
                    ),
                    html.Label('Y-axis parameter'),
                    dcc.Dropdown(
                        self.float_parameters_names,
                        self.float_parameters_names[1],
                        id='crossfilter-yaxis-column-1',
                    ),
                ], style={'width': '20%'}),
                html.Div(children=[
                    dcc.Graph(
                        id='crossfilter-indicator-scatter-1',
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '75%', 'height': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '90%'}),
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def render_objective_function_value_scatter(self):
        return html.Div(children=[
            html.H2('Scatter of Objective Function Values',
                    style={'textAlign': 'left'}),

            dcc.Tabs([
                dcc.Tab(label='Float Parameters', children=[
                    html.Div([
                        html.Div(children=[
                            html.Br(),
                            html.Label('X-axis parameter'),
                            dcc.Dropdown(
                                self.float_parameters_names,
                                self.float_parameters_names[0],
                                id='crossfilter-xaxis-column',
                            ),
                        ], style={'width': '20%'}),
                        html.Div(children=[
                            dcc.Graph(
                                id='crossfilter-indicator-scatter',
                                config={
                                    'displayModeBar': True,  # True, False, 'hover'
                                },
                            )
                        ], style={'width': '75%'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '70%'}),
                ]),
                dcc.Tab(label='Discrete Parameters', children=[
                    html.Div([
                        html.Div(children=[
                            html.Br(),
                            html.Label('X-axis parameter'),
                            dcc.Dropdown(
                                self.discrete_parameters_names,
                                self.discrete_parameters_names[0],
                                id='crossfilter-xaxis-column-2',
                            ),
                        ], style={'width': '20%'}),
                        html.Div(children=[
                            dcc.Graph(
                                id='crossfilter-indicator-scatter-2',
                                config={
                                    'displayModeBar': True,  # True, False, 'hover'
                                },
                            )
                        ], style={'width': '75%'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '70%'}),
                ])
            ], style={'width': '90%'})
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def render_page_content(self, pathname):
        if pathname == "/":
            return [
                html.Div([
                    self.render_parameters_discription(),
                    self.render_task_discription(),
                ], style={'display': 'flex', 'flexDirection': 'row'}),

                html.Div([
                    self.render_solution_discription(),
                    self.render_optimization_time_plot()
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                self.render_iteration_characteristic(),
            ]

        elif pathname == "/page-2":
            if self.mode == 'Release':
                return html.Div(children=[
                    html.H1('All Trials Archive',
                            style={'textAlign': 'left'}),
                    generate_table(self.dfSDI[['trial'] + self.parameters_names + ["__z", "creation_time"]]),
                    html.Div(id='datatable-interactivity-container', style={'width': '95%'})
                ], style = {"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })
            elif self.mode == 'Debug':
                return html.Div(children=[
                    html.H1('All Trials Archive',
                            style={'textAlign': 'left'}),
                    generate_table(self.dfSDI),
                    html.Div(id='datatable-interactivity-container', style={'width': '95%'})
                ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })
        elif pathname == "/page-1":
            return [
                self.render_multidimentional_visualization(),
                self.render_3D_graph(),
                self.render_objective_function_value_scatter(),

                html.Div([
                    self.render_dependence(),
                    self.render_importance()
                ], style={'display': 'flex', 'flexDirection': 'row'}),
            ]

        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

    def update_graph_1(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.discrete_parameters_names[0]

        fig = px.violin(self.dfSDI,
                         x=xaxis_column_name, y='__z',
                         title="Scatter of objective function values by selected parameter",
                         color_discrete_sequence=['#3E59A5']
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        return fig

    def update_graph_2(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]

        fig = px.scatter(self.dfSDI,
                         x=xaxis_column_name,
                         y='__z',
                         color=self.dfSDI['trial'][::-1],
                         color_continuous_scale="Ice",
                         title="Scatter of objective function values by selected parameter",
                         opacity=0.7
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        return fig

    def update_graph_3(self, xaxis_column_name=None, yaxis_column_name=None, type='3D Surface', calc='interpolation'):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]
        if yaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[1]

        df = self.dfSDI

        # использовать окрестность лучшего решения

        # берем окрестность лучшего сочетания дискретных параметров
        for param in self.discrete_parameters_names:
            df = df.loc[df[param] == self.best_discrete_point_dictionary[param]]

        '''
        # берем окрестность лучших прочих непрерывных параметров
        for param in self.float_parameters_names:
            if (xaxis_column_name != param and yaxis_column_name != param):
                df = df.loc[abs(df[param] - self.best_float_point_dictionary[param]) < self.parameter_eps]
        '''

        x = np.array(df[xaxis_column_name].values)
        y = np.array(df[yaxis_column_name].values)
        z = np.array(df['__z'].values)

        bounds_x = list(self.float_parameters_bounds[(self.float_parameters_names).index(xaxis_column_name)].values())[
            0]
        bounds_y = list(self.float_parameters_bounds[(self.float_parameters_names).index(yaxis_column_name)].values())[
            0]

        if calc == 'interpolation':
            xi = np.linspace(bounds_x[0], bounds_x[1], 150)
            yi = np.linspace(bounds_y[0], bounds_y[1], 150)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x, y), z, (X, Y), method='cubic')  # "nearest", "linear", "natural", and "cubic" methods
        elif calc == 'approximation':
            nn = MLPRegressor(activation='logistic',  # can be tanh, identity, logistic, relu
                              solver='lbfgs',  # can be lbfgs, sgd , adam
                              alpha=0.001,
                              hidden_layer_sizes=(40,),
                              max_iter=10000,
                              tol=10e-6,
                              random_state=10)

            points = [list(x), list(y)]
            points = list(map(list, zip(*points)))

            nn.fit(points, z)
            xi = np.linspace(bounds_x[0], bounds_x[1], 150)
            yi = np.linspace(bounds_y[0], bounds_y[1], 150)
            X, Y = np.meshgrid(xi, yi)

            xy = np.c_[X.ravel(), Y.ravel()]

            Z = nn.predict(xy)
            Z = Z.reshape(150, 150)
        elif calc == 'none (by trials points)':
            pass

        if type == '3D Surface':
            if calc == 'interpolation' or calc == 'approximation':
                surface = go.Surface(x=xi, y=yi, z=Z, colorscale="Ice", opacity=1)
                fig = go.Figure(data=[surface])
                fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                                  highlightcolor="limegreen", project_z=True))
            elif calc == 'none (by trials points)':
                surface = go.Mesh3d(x=x, y=y, z=z, showscale=True, intensity=z, colorscale="Ice", opacity=1)
                fig = go.Figure(data=[surface])

            fig.add_scatter3d(x=x, y=y, z=z, mode='markers', name='trials points', marker=dict(size=2, color='red', opacity=0.7))
            fig.add_scatter3d(x=[self.best_float_point_dictionary[xaxis_column_name]],
                              y=[self.best_float_point_dictionary[yaxis_column_name]],
                              z=[self.best_value], name='best trial point',
                              mode='markers', marker=dict(size=3, color='green', opacity=1))
            fig.update_layout(title='3D Surface', paper_bgcolor='#FFFFFF',
                              plot_bgcolor='#FFFFFF', showlegend=False, height=700,
                              template="none")
        else:
            if calc == 'interpolation' or calc == 'approximation':
                fig = go.Figure(data=[go.Contour(x=xi, y=yi, z=Z, colorscale="Ice",
                                                 colorbar=dict(title='objective function values',
                                                 titleside='right'))])
            elif calc == 'none (by trials points)':
                fig = go.Figure(data=[go.Contour(x=x, y=y, z=z, colorscale="Ice",
                                                 colorbar=dict(title='objective function values',
                                                 titleside='right'))])  # , line_smoothing=0.85

            fig.add_scatter(x=x, y=y, mode='markers', name='trials points', marker=dict(size=2, color='red', opacity=0.7))
            fig.add_scatter(x=[self.best_float_point_dictionary[xaxis_column_name]],
                            y=[self.best_float_point_dictionary[yaxis_column_name]],
                            mode='markers', name='best trial point', marker=dict(size=4, color='green', opacity=1))
            fig.add_trace(go.Histogram(
                y=y,
                xaxis='x2',
                marker=dict(
                    color='#3E59A5'
                ),
                name=yaxis_column_name+' values histogram'
            ))
            fig.add_trace(go.Histogram(
                x=x,
                yaxis='y2',
                marker=dict(
                    color='#3E59A5'
                ),
                name=xaxis_column_name+' values histogram'
            ))
            fig.update_layout(title='Levels Lines', paper_bgcolor='#FFFFFF',
                              plot_bgcolor='#FFFFFF', showlegend=True, height=700,
                              template="none",legend={'orientation': "h"},
                              xaxis_showgrid=False, yaxis_showgrid=False,
                              xaxis_range=[bounds_x[0], bounds_x[1]], yaxis_range=[bounds_y[0], bounds_y[1]],
                              xaxis_domain=[0, 0.85],
                              yaxis_domain=[0, 0.85],
                              xaxis2=dict(
                                  zeroline=False,
                                  domain=[0.85, 1],
                                  showgrid=False
                              ),
                              yaxis2=dict(
                                  zeroline=False,
                                  domain=[0.85, 1],
                                  showgrid=False
                              ),
                              bargap=0,
                              hovermode='closest',
            )

        return fig

    def update_graph_4(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.parameters_names
        xaxis_column_name = ['__z'] + xaxis_column_name

        '''
        df = self.normalizate_df[self.normalizate_df['param_name'].isin(xaxis_column_name)]

        fig = px.line(
            df, x='param_name', y='norm_value', color='trial', hover_data='actual_value',
            title="Scatter of objective function values by one parameter",
            markers=True,
            color_discrete_map=self.value_colors_dictionary
        )

        fig.update_xaxes(title='parameters')
        fig.update_yaxes(title='objective function')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        '''

        xaxis_column_name_dict = {}
        for name in xaxis_column_name:
            replace = ''
            if name in self.discrete_parameters_names:
                self.dfSDI[name+'_cat'] = self.dfSDI[name].astype('category').cat.codes
                replace = name+'_cat'
            else:
                replace = name
            xaxis_column_name_dict[replace] = name


        fig = px.parallel_coordinates(self.dfSDI, color="__z",
                                      dimensions=xaxis_column_name_dict.keys(),
                                      labels=xaxis_column_name_dict,
                                      color_continuous_scale='ice')

        fig.update_layout(
            xaxis=dict(
                title='parameters',
                ticktext=xaxis_column_name
            ),
            yaxis=dict(
                title='objective function'
            ),
            paper_bgcolor = '#FFFFFF', plot_bgcolor = '#FFFFFF'
        )

        fig.update_traces(unselected_line_opacity=0.5, selector=dict(type='parcoords'))

        return fig

    def update_output(self, contents):
        if self.init and contents is not None:
            content_type, content_string = contents[0].split(',')
            decoded = base64.b64decode(content_string)
            self.data = json.loads(decoded)
            self.launch()
        else:
            self.init = True
        return html.Div(id='hidden-div', style={'display':'none'})

    def update_arhive_graph(self, rows, derived_virtual_selected_rows):
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows = []

        dff = self.dfSDI if rows is None else pd.DataFrame(rows)
        colors = ['#31B37C' if i in derived_virtual_selected_rows else '#3E59A5'
                  for i in range(len(dff))]

        return [
            dcc.Graph(
                figure={
                    "data": [
                        {
                            "x": dff['trial'],
                            "y": dff['__z'],
                            "type": "bar",
                            "marker": {"color": colors},
                        }
                    ],
                    "layout": {
                        "xaxis": {"automargin": True, "title" : "trial"},
                        "yaxis": {
                            "automargin": True,
                            "title": {"text": "objective function"}
                        },
                        "height": 250,
                        "weight": "95%",
                        "margin": {"t": 10, "l": 10, "r": 10},
                        "paper_bgcolor":'#FFFFFF',
                        "plot_bgcolor":'#FFFFFF'
                    },
                },
            )
        ]