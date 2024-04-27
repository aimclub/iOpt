import dash
from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from scipy.interpolate import griddata
from sklearn.neural_network import MLPRegressor
import dash_daq as daq

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

def create_dash_table_from_dataframe(dataframe):
    return html.Div([dash_table.DataTable(
        dataframe.to_dict('records'), [{"name": i, "id": i} for i in dataframe.columns],
        filter_action="native", id='datatable-interactivity', sort_action="native",
        row_selectable="multi", selected_rows=[],
        sort_mode="multi", page_size=12)],
        style={'maxWidth': '95%', 'maxHeight': '700px', "overflow": "scroll"}
    )

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
            Output('discrete_scatter_figure', 'figure'),
            Input('crossfilter-xaxis-column-2', 'value')
        )(self.update_discrete_scatter_figure)

        self.app.callback(
            Output('continuous_scatter_figure', 'figure'),
            Input('crossfilter-xaxis-column', 'value')
        )(self.update_continuous_scatter_figure)

        self.app.callback(
            Output('surface_or_lines_level_figure', 'figure'),
            Input('crossfilter-xaxis-column-1', 'value'),
            Input('crossfilter-yaxis-column-1', 'value'),
            Input('crossfilter-type-1', 'value'),
            Input('crossfilter-type-2', 'value'),
            Input('my-boolean-switch', 'on'),
        )(self.update_surface_or_lines_level_figure)

        self.app.callback(
            Output('multidimensional_figure', 'figure'),
            Input('crossfilter-xaxis-column-3', 'value')
        )(self.update_multidimensional_figure)

        self.app.callback(
            Output('data_from_json_file', 'children'),
            Input('upload-data', 'contents')
        )(self.read_data_from_json)

        self.app.callback(
            Output('archive_figure', "children"),
            Input('datatable-interactivity', "derived_virtual_data"),
            Input('datatable-interactivity', "derived_virtual_selected_rows")
        )(self.update_archive_figure)

    def read_and_prepare_data(self, data):
        self.parameter_eps = data['Parameters'][0]['eps']
        self.parameter_r = data['Parameters'][0]['r']
        self.parameter_iters_limit = data['Parameters'][0]['iters_limit']
        self.task_name = data['Task'][0]['name']

        self.functional_count = len(data['SearchDataItem'][0]["function_values"])

        self.dfSDI = pd.DataFrame(data['SearchDataItem'])
        self.dfSDI = self.dfSDI.rename(columns={'__z': 'objective_func'})
        dfFV = pd.DataFrame(data['Task'][0]['float_variables'])
        dfDV = pd.DataFrame(data['Task'][0]['discrete_variables'])
        dfBT = pd.DataFrame(data['best_trials'])
        dfBT = dfBT.rename(columns={'__z': 'objective_func'})
        dfS = pd.DataFrame(data['solution'])

        self.float_parameters_bounds = data['Task'][0]['float_variables']
        self.discrete_parameters_valid_values = data['Task'][0]['discrete_variables']

        self.float_parameters_names = [f"{x} [c]" for x in [*dfFV]]
        self.discrete_parameters_names = [f"{x} [d]" for x in [*dfDV]]

        self.parameters_names = self.float_parameters_names + self.discrete_parameters_names

        self.best_value = dfBT['objective_func'][0]
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

        self.dfSDI = self.dfSDI[["objective_func", "creation_time", "x", "delta", "globalR", "localR"]]
        self.dfSDI = pd.concat([DVs, self.dfSDI], axis=1, join='inner')
        self.dfSDI = pd.concat([FVs, self.dfSDI], axis=1, join='inner')

        boundary_points_indexes = self.dfSDI[self.dfSDI['objective_func'] >= 1.797692e+308].index
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
        self.calculate_parameters_importance()

        del self.data
        self.data = None


    def calculate_parameters_importance(self):
        minZ = min(self.dfSDI['objective_func'])
        maxZ = max(self.dfSDI['objective_func'])

        self.importance = []
        for parameter_name in self.parameters_names:
            uniq_parameter_values = self.dfSDI[parameter_name].unique()
            parameter_importance_value = 0
            for value in uniq_parameter_values:
                data = self.dfSDI.loc[self.dfSDI[parameter_name] == value]
                parameter_importance_value += (max(data['objective_func']) - min(data['objective_func']))
            self.importance.append(round(parameter_importance_value / (len(uniq_parameter_values) * (maxZ - minZ)), 2))

    def create_sidebar_navigator(self):
        self.sidebar = html.Div(
            [
                dbc.Navbar(dbc.Container([dbc.Col([dbc.Nav([
                        dbc.NavItem(html.Img(src=r'assets/iOptdash_light_.png', alt='image', height="50px")),
                        dbc.NavItem(dbc.NavLink("Solution", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", active="exact")),
                        dbc.NavItem(
                            dbc.NavLink("Archive", href="/archive", active="exact"),
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
            ]
        )

    def create_content_page(self):
        self.content = html.Div(id="page-content", style=CONTENT_STYLE)

    def launch(self):
        self.read_and_prepare_data(self.data)

        self.create_sidebar_navigator()
        self.create_content_page()

        self.app.layout = html.Div([
            dcc.Location(id="url"),
            self.sidebar,
            self.content
        ], style={"background-color": "#FBFBFB"})

        self.app.run(debug=True)

    def render_problem_description(self):
        return html.Div(children=[
            html.H2('PROBLEM DESCRIPTION',
                    style={'textAlign': 'left'}),
            html.P(f"Problem name: {self.task_name}", style={'color': '#212121'}),
            html.P(f"Number of functionalities: objective = 1, constraint = {self.functional_count - 1}", style={'color': '#212121'}),
            html.P(f"Number of parameters: continuous = {len(self.float_parameters_names)}, discrete = {len(self.discrete_parameters_names)}", style={'color': '#212121'}),
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
            html.Div(id='data_from_json_file')
            ],
            style={'width': '55%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", }
        )
    def render_parameters_description(self):
        return html.Div(children=[
            html.H2('PARAMETERS USED',
                    style={'textAlign': 'left'}),
            html.P(f"Required accuracy: eps = {self.parameter_eps}",
                   style={'color': '#212121'}),
            html.P(f"Reliability parameter: r = {self.parameter_r}",
                   style={'color': '#212121'}),
            html.P(f"Limiting the number of iterations: iters_limit = {self.parameter_iters_limit}",
                   style={'color': '#212121'}),
            ],
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", }
        )
    # "border": "3px #FFFFFF outset"
    def render_solution_description(self):
        return html.Div(children=[
            html.H2('Solution',
                    style={'textAlign': 'left'}),
            html.P(
                f"Total trials number: {self.trial_number} (global trials number - {self.global_iter_number}, local trials number - {self.local_iter_number})",
                style={'color': '#212121'}),
            html.P(f"{round(self.best_value, 6)}", style={'font-size': '2.0em', 'color': 'black'}),
            html.P(f"Best point: {self.best_float_point_dictionary}, {self.best_discrete_point_dictionary}",
                   style={'color': '#212121'}),
            html.P(f'Best trial number: {self.best_trial_number}', style={'color': '#212121'}),
            html.P(f"Accuracy: {round(self.accuracy, 6)}", style={'color': '#212121'}),
            html.P(f"*[c] - continuous variables, [d] - discrete variables",
                   style={'color': '#212121', 'font-size': '0.8em'}),
        ], className="best trial description",
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def render_optimization_time(self):
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
                        figure=((px.scatter(self.dfSDI, x='trial', y='objective_func', color_discrete_sequence=['#3E59A5'], marginal_y="histogram", trendline="expanding", trendline_options=dict(function="min"))).update_layout(legend= {'orientation': "h", 'y': -0.25},
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

    def scatter_matrix_figure(self):
        fig = px.scatter_matrix(self.dfSDI, dimensions=self.parameters_names, color="objective_func", opacity=0.7,
                                color_continuous_scale="Ice", width=len(self.parameters_names) * 200,
                                height=len(self.parameters_names) * 200)
        fig.update_traces(diagonal_visible=False)
        '''
        fig = ff.create_scatterplotmatrix(self.dfSDI[self.parameters_names+['objective_func']], diag='histogram', index='objective_func',
                                          opacity=0.9, colormap=px.colors.sequential.ice, width=len(self.parameters_names) * 200,
                                          height=len(self.parameters_names) * 200)
        '''

        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        return fig

    def render_parameters_dependence(self):
        return html.Div([
            html.H2('Parameters Dependence',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure=self.scatter_matrix_figure(),
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '90%'}),
            ], style={'maxWidth': '90%', 'maxHeight': '650px', "overflow": "scroll"}),
        ], style={'width': '65%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", 'height': '750px'})

    def render_parameters_importance(self):
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

    def render_multidimensional_representation(self):
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
                        id='multidimensional_figure',
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '85%'}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'height': '90%'})
        ], style={ "background-color": "#FFFFFF", "border": "20px solid #FFFFFF"})

    def render_surface_and_level_lines(self):
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
                    html.Label('Y-axis parameter'),
                    dcc.Dropdown(
                        self.float_parameters_names,
                        self.float_parameters_names[1],
                        id='crossfilter-yaxis-column-1',
                    ),
                    html.Label('Show parameters bars'),
                    html.Td(
                        daq.BooleanSwitch(id='my-boolean-switch', on=True, color="#3E59A5")
                    )
                ], style={'width': '20%'}),
                html.Div(children=[
                    dcc.Graph(
                        id='surface_or_lines_level_figure',
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '75%', 'height': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '90%'}),
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def render_objective_function_values_scatter(self):
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
                                id='continuous_scatter_figure',
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
                                id='discrete_scatter_figure',
                                config={
                                    'displayModeBar': True,  # True, False, 'hover'
                                },
                            )
                        ], style={'width': '75%'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '70%'}),
                ])
            ], style={'width': '90%'})
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })
    def render_release_archive(self):
        return html.Div(children=[
            html.P(f"*[c] - continuous variables, [d] - discrete variables",
                   style={'text-align': 'right', 'color': '#212121', 'font-size': '0.8em'}),
            html.H1('All Trials Archive',
                    style={'textAlign': 'left'}),
            create_dash_table_from_dataframe(self.dfSDI[['trial'] + self.parameters_names + ["objective_func", "creation_time"]]),
            html.Div(id='archive_figure', style={'width': '95%'})
        ], style = {"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })
    def render_debug_archive(self):
        return html.Div(children=[
            html.P(f"*[c] - continuous variables, [d] - discrete variables",
                   style={'text-align': 'right', 'color': '#212121', 'font-size': '0.8em'}),
            html.H1('All Trials Archive',
                    style={'textAlign': 'left'}),
            create_dash_table_from_dataframe(self.dfSDI),
            html.Div(id='archive_figure', style={'width': '95%'})
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })
    def render_page_content(self, pathname):
        if pathname == "/":
            return [
                html.Div([
                    self.render_parameters_description(),
                    self.render_problem_description(),
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                html.Div([
                    self.render_solution_description(),
                    self.render_optimization_time()
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                self.render_iteration_characteristic(),
            ]

        elif pathname == "/archive":
            if self.mode == 'Release':
                return [
                    self.render_release_archive()
                ]
            elif self.mode == 'Debug':
                return [
                    self.render_debug_archive()
                ]
        elif pathname == "/analytics":
            return [
                html.Div(children=[
                    html.P(f"*[c] - continuous variables, [d] - discrete variables",
                       style={'text-align': 'right', 'color': '#212121', 'font-size': '0.8em'}),
                ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF"}),
                self.render_multidimensional_representation(),
                self.render_surface_and_level_lines(),
                self.render_objective_function_values_scatter(),
                html.Div([
                    self.render_parameters_dependence(),
                    self.render_parameters_importance()
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

    def update_discrete_scatter_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.discrete_parameters_names[0]

        fig = px.violin(self.dfSDI,
                         x=xaxis_column_name, y='objective_func',
                         title="Scatter of objective function values by selected parameter",
                         color_discrete_sequence=['#3E59A5']
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        return fig

    def update_continuous_scatter_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]

        fig = px.scatter(self.dfSDI,
                         x=xaxis_column_name,
                         y='objective_func',
                         color=self.dfSDI['trial'][::-1],
                         color_continuous_scale="Ice",
                         title="Scatter of objective function values by selected parameter",
                         opacity=0.7
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        return fig

    def calculate_data(self, xaxis_column_name, yaxis_column_name, calc):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]
        if yaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[1]

        df = None

        # берем окрестность лучшего сочетания дискретных параметров
        for param in self.discrete_parameters_names:
            df = self.dfSDI.loc[self.dfSDI[param] == self.best_discrete_point_dictionary[param]]

        '''
        # берем окрестность лучших прочих непрерывных параметров
        for param in self.float_parameters_names:
            if (xaxis_column_name != param and yaxis_column_name != param):
                df = df.loc[abs(df[param] - self.best_float_point_dictionary[param]) < self.parameter_eps]
        '''

        x = np.array(df[xaxis_column_name].values)
        y = np.array(df[yaxis_column_name].values)
        z = np.array(df['objective_func'].values)

        xi = None
        yi = None
        Z = None

        bounds_x = list(self.float_parameters_bounds[(self.float_parameters_names).index(xaxis_column_name)].values())[0]
        bounds_y = list(self.float_parameters_bounds[(self.float_parameters_names).index(yaxis_column_name)].values())[0]

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

        return bounds_x, bounds_y, x, y, z, xi, yi, Z

    def surface_figure(self, xaxis_column_name, yaxis_column_name, calc, xi, yi, Z, x, y, z):
        if calc == 'interpolation' or calc == 'approximation':
            surface = go.Surface(x=xi, y=yi, z=Z, colorscale="Ice", opacity=1)
            fig = go.Figure(data=[surface])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                              highlightcolor="limegreen", project_z=True))
        elif calc == 'none (by trials points)':
            surface = go.Mesh3d(x=x, y=y, z=z, showscale=True, intensity=z, colorscale="Ice", opacity=1)
            fig = go.Figure(data=[surface])

        fig.add_scatter3d(x=x, y=y, z=z, mode='markers', name='trials points',
                          marker=dict(size=2, color='red', opacity=0.7))
        fig.add_scatter3d(x=[self.best_float_point_dictionary[xaxis_column_name]],
                          y=[self.best_float_point_dictionary[yaxis_column_name]],
                          z=[self.best_value], name='best trial point',
                          mode='markers', marker=dict(size=3, color='green', opacity=1))
        fig.update_layout(scene=dict(xaxis_title=xaxis_column_name, yaxis_title=yaxis_column_name,
                          zaxis_title='objective function'), title='', paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#FFFFFF', showlegend=False, height=700,
                          template="none")
        return fig

    def lines_level_figure(self, xaxis_column_name, yaxis_column_name, calc, showBars, bounds_x, bounds_y, xi, yi, Z, x, y, z):
        if calc == 'interpolation' or calc == 'approximation':
            fig = go.Figure(data=[go.Contour(x=xi, y=yi, z=Z, colorscale="Ice",
                            colorbar=dict(title='objective function', titleside='right'))])
        elif calc == 'none (by trials points)':
            fig = go.Figure(data=[go.Contour(x=x, y=y, z=z, colorscale="Ice",
                            colorbar=dict(title='objective function', titleside='right'))])

        fig.add_scatter(x=x, y=y, mode='markers', name='trials points', marker=dict(size=2, color='red', opacity=0.7))
        fig.add_scatter(x=[self.best_float_point_dictionary[xaxis_column_name]],
                        y=[self.best_float_point_dictionary[yaxis_column_name]],
                        mode='markers', name='best trial point', marker=dict(size=4, color='green', opacity=1))

        fig.update_layout(title='', paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#FFFFFF', showlegend=True, height=700,
                          legend={'orientation': "h"},
                          xaxis_range=[bounds_x[0], bounds_x[1]], yaxis_range=[bounds_y[0], bounds_y[1]],
                          xaxis_title=xaxis_column_name,
                          yaxis_title=yaxis_column_name)
        if showBars:
            fig.add_trace(go.Histogram(
                y=y,
                xaxis='x2',
                marker=dict(
                    color='#3E59A5'
                ),
                name=yaxis_column_name + ' values histogram'
            ))
            fig.add_trace(go.Histogram(
                x=x,
                yaxis='y2',
                marker=dict(
                    color='#3E59A5'
                ),
                name=xaxis_column_name + ' values histogram'
            ))

            fig.update_layout(
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
    def update_surface_or_lines_level_figure(self, xaxis_column_name=None, yaxis_column_name=None, type='3D Surface', calc='interpolation', showBars=True):
        bounds_x, bounds_y, x, y, z, xi, yi, Z = self.calculate_data(xaxis_column_name, yaxis_column_name, calc)
        if type == '3D Surface':
            return self.surface_figure(xaxis_column_name, yaxis_column_name, calc, xi, yi, Z, x, y, z)
        elif type == 'Heatmap':
            return self.lines_level_figure(xaxis_column_name, yaxis_column_name, calc, showBars, bounds_x, bounds_y, xi, yi, Z, x, y, z)

    def update_multidimensional_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.parameters_names
        xaxis_column_name = ['objective_func'] + xaxis_column_name

        xaxis_column_name_dict = {}
        for name in xaxis_column_name:
            replace = ''
            if name in self.discrete_parameters_names:
                self.dfSDI[name+'_cat'] = self.dfSDI[name].astype('category').cat.codes
                replace = name+'_cat'
            else:
                replace = name
            xaxis_column_name_dict[replace] = name


        fig = px.parallel_coordinates(self.dfSDI, color="objective_func",
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

    def read_data_from_json(self, contents):
        if self.init and contents is not None:
            content_type, content_string = contents[0].split(',')
            decoded = base64.b64decode(content_string)
            self.data = json.loads(decoded)
            self.launch()
        else:
            self.init = True
        return html.Div(id='hidden-div', style={'display':'none'})

    def update_archive_figure(self, rows, derived_virtual_selected_rows):
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
                            "y": dff['objective_func'],
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