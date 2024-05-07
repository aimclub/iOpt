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
        self.__app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self.__app.config['suppress_callback_exceptions'] = True
        self.__mode = mode
        self.__data_path = data_path
        with open(self.__data_path) as json_data:
            self.__data = json.load(json_data)
        self.__init = False
        self.__color_themes = ["Ice", "Viridis", "Bluered", "RdBu_r", "Jet", "Electric", "Plasma"]
        self.__color_themes_dict = {"Ice" : px.colors.sequential.ice,
                                  "Viridis" : px.colors.sequential.Viridis,
                                  "Bluered" : px.colors.sequential.Bluered,
                                  "Electric" : px.colors.sequential.Electric,
                                  "Jet": px.colors.sequential.Jet,
                                  "Plasma": px.colors.sequential.Plasma,
                                  "RdBu_r": px.colors.sequential.RdBu_r
                                    }
        self.__choosen_color = px.colors.sequential.ice
        self.__base_color = px.colors.sequential.ice[0]
        self.__task_name = None
        self.__dfSDI = None
        self.__best_value = None
        self.__float_parameters_names = None
        self.__trial_number = None
        self.__float_parameters_bounds = None
        self.__discrete_parameters_names = None
        self.__discrete_parameters_valid_values = None
        self.__best_trial_number = None
        self.__trial_names = None
        self.__values_update_best = None
        self.__points_update_best = None
        self.__importance = None
        self.__sidebar = None
        self.__content = None
        self.__global_iter_number = None
        self.__local_iter_number = None
        self.__accuracy = None
        self.__solve_time = None
        self.__best_float_point_dictionary = None
        self.__best_discrete_point_dictionary = None
        self.__optimization_time = None
        self.__value_colors_dictionary = None
        self.__values_names_T = None
        self.__parameter_eps = None
        self.__parameter_r = None
        self.__parameter_iters_limit = None
        self.__values_T_dictionary = None

        self.__obj_func_norm = None

        self.__functional_count = 1

        self.__dfSDI_original = None

        self.__app.callback(
            Output("page-content", "children"),
            Input("url", "pathname"),
            Input('crossfilter-xaxis-column-545757', "value"),
            Input('my-boolean-switch-2', 'on')
        )(self.__render_page_content)

        self.__app.callback(
            Output('discrete_scatter_figure', 'figure'),
            Input('crossfilter-xaxis-column-2', 'value')
        )(self.__update_discrete_scatter_figure)

        self.__app.callback(
            Output('continuous_scatter_figure', 'figure'),
            Input('crossfilter-xaxis-column', 'value')
        )(self.__update_continuous_scatter_figure)

        self.__app.callback(
            Output('surface_or_lines_level_figure', 'figure'),
            Input('crossfilter-xaxis-column-1', 'value'),
            Input('crossfilter-yaxis-column-1', 'value'),
            Input('crossfilter-type-1', 'value'),
            Input('crossfilter-type-2', 'value'),
            Input('my-boolean-switch', 'on'),
        )(self.__update_surface_or_lines_level_figure)

        self.__app.callback(
            Output('multidimensional_figure', 'figure'),
            Input('crossfilter-xaxis-column-3', 'value')
        )(self.__update_multidimensional_figure)

        self.__app.callback(
            Output('data_from_json_file', 'children'),
            Input('upload-data', 'contents')
        )(self.__read_data_from_json)

        self.__app.callback(
            Output('archive_figure', "children"),
            Input('datatable-interactivity', "derived_virtual_data"),
            Input('datatable-interactivity', "derived_virtual_selected_rows")
        )(self.__update_archive_figure)

    def launch(self):
        self.__read_and_prepare_data(self.__data)

        self.__create_sidebar_navigator()
        self.__create_content_page()

        self.__app.layout = html.Div([
            dcc.Location(id="url"),
            self.__sidebar,
            self.__content
        ], style={"background-color": "#FBFBFB"})

        self.__app.run(debug=True)

    def __read_and_prepare_data(self, data):
        self.__parameter_eps = data['Parameters'][0]['eps']
        self.__parameter_r = data['Parameters'][0]['r']
        self.__parameter_iters_limit = data['Parameters'][0]['iters_limit']
        self.__task_name = data['Task'][0]['name']

        self.__functional_count = len(data['SearchDataItem'][0]["function_values"])

        self.__dfSDI = pd.DataFrame(data['SearchDataItem'])
        self.__dfSDI = self.__dfSDI.rename(columns={'__z': 'objective_func'})
        dfFV = pd.DataFrame(data['Task'][0]['float_variables'])
        dfDV = pd.DataFrame(data['Task'][0]['discrete_variables'])
        dfBT = pd.DataFrame(data['best_trials'])
        dfBT = dfBT.rename(columns={'__z': 'objective_func'})
        dfS = pd.DataFrame(data['solution'])

        self.__float_parameters_bounds = data['Task'][0]['float_variables']
        self.__discrete_parameters_valid_values = data['Task'][0]['discrete_variables']

        self.__float_parameters_names = [f"{x} [c]" for x in [*dfFV]]
        self.__discrete_parameters_names = [f"{x} [d]" for x in [*dfDV]]

        self.parameters_names = self.__float_parameters_names + self.__discrete_parameters_names

        self.__best_value = dfBT['objective_func'][0]
        self.__accuracy = float(dfS['solution_accuracy'])
        self.__trial_number = int(dfS['number_of_trials'])
        self.__global_iter_number = int(dfS['number_of_global_trials'])
        self.__local_iter_number = int(dfS['number_of_local_trials'])
        self.__solve_time = float(dfS['solving_time'])
        self.__best_trial_number = int(dfS['num_iteration_best_trial'][0][0])

        FVs = pd.DataFrame(self.__dfSDI['float_variables'].to_list(), columns=self.__float_parameters_names,
                           index=self.__dfSDI.index)
        DVs = pd.DataFrame(self.__dfSDI['discrete_variables'].to_list(), columns=self.__discrete_parameters_names,
                           index=self.__dfSDI.index)

        self.__optimization_time = self.__dfSDI["creation_time"].to_list()

        isFirst = 1
        shift = 0
        new_times = []
        for elem in self.__optimization_time:
            if elem != 0 and isFirst:
                shift = elem
                isFirst = 0
            if elem != 0:
                elem -= shift
            new_times.append(elem)

        self.__optimization_time = new_times

        self.__dfSDI = self.__dfSDI[["objective_func", "x", "delta", "globalR", "localR"]]
        self.__dfSDI = pd.concat([DVs, self.__dfSDI], axis=1, join='inner')
        self.__dfSDI = pd.concat([FVs, self.__dfSDI], axis=1, join='inner')

        boundary_points_indexes = self.__dfSDI[self.__dfSDI['objective_func'] >= 1.797692e+308].index
        self.__dfSDI.drop(boundary_points_indexes, inplace=True)
        self.__dfSDI.insert(loc=0, column='trial', value=np.arange(1, len(self.__dfSDI) + 1))

        accuracy_in_signs = 0
        accuracy_copy = self.__accuracy
        while accuracy_copy < 1:
            accuracy_copy *= 10
            accuracy_in_signs += 1

        self.__best_float_point_dictionary = dict(
            zip(self.__float_parameters_names, [round(elem, accuracy_in_signs) for elem in dfBT['float_variables'][0]]))
        self.__best_discrete_point_dictionary = dict(zip(self.__discrete_parameters_names, dfBT['discrete_variables'][0]))



        self.__trial_names = ['trial ' + str(index + 1) for index in range(self.__trial_number)]
        self.__calculate_parameters_importance()

        del self.__data
        self.__data = None

        self.__dfSDI_original = self.__dfSDI

    def __calculate_parameters_importance(self):
        minZ = min(self.__dfSDI['objective_func'])
        maxZ = max(self.__dfSDI['objective_func'])

        self.__importance = []
        for parameter_name in self.parameters_names:
            uniq_parameter_values = self.__dfSDI[parameter_name].unique()
            parameter_importance_value = 0
            for value in uniq_parameter_values:
                data = self.__dfSDI.loc[self.__dfSDI[parameter_name] == value]
                parameter_importance_value += (max(data['objective_func']) - min(data['objective_func']))
            self.__importance.append(round(parameter_importance_value / (len(uniq_parameter_values) * (maxZ - minZ)), 2))

    def __create_sidebar_navigator(self):
        self.__sidebar = html.Div(
            [
                dbc.Navbar(dbc.Container([dbc.Col([dbc.Nav([
                        dbc.NavItem(html.Img(src=r'assets/iOptdash_light_.png', alt='image', height="50px")),
                        dbc.NavItem(dbc.NavLink("Solution", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", active="exact")),
                        dbc.NavItem(dbc.NavLink("Archive", href="/archive", active="exact"), className="me-auto"),

                        dbc.NavItem(dbc.NavLink("About", href="https://iopt.readthedocs.io/ru/latest")),
                        dbc.NavItem(dbc.NavLink("Github Source", href="https://github.com/aimclub/iOpt")),

                        html.Div(
                            [
                                html.P(f"Eliminate outliers", style={'color': '#989898', 'font-size': '0.8em'}),
                                html.Td(
                                    daq.BooleanSwitch(id='my-boolean-switch-2', on=False, color='red')
                                ),
                            ], style={'width': '12%', 'position': 'right'}
                        ),

                        html.Div(
                            [
                                html.P(f"Colorscheme", style={'color': '#989898', 'font-size': '0.8em'}),
                                dcc.Dropdown(
                                    self.__color_themes,
                                    "Ice",
                                    id='crossfilter-xaxis-column-545757'
                                ),
                            ], style={'width': '12%','position': 'right'}
                        ),
                    ]),
                    ]
                )]),
                    sticky="bottom",
                    color="primary",
                    dark=True,
                ),
            ]
        )

    def __create_content_page(self):
        self.__content = html.Div(id="page-content", style=CONTENT_STYLE)

    def __render_problem_description(self):
        return html.Div(children=[
            html.H2('PROBLEM DESCRIPTION',
                    style={'textAlign': 'left'}),
            html.P(f"Problem name: {self.__task_name}", style={'color': '#212121'}),
            html.P(f"Number of functionalities: objective = 1, constraint = {self.__functional_count - 1}", style={'color': '#212121'}),
            html.P(f"Number of parameters: continuous = {len(self.__float_parameters_names)}, discrete = {len(self.__discrete_parameters_names)}", style={'color': '#212121'}),
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

    def __render_parameters_description(self):
        return html.Div(children=[
            html.H2('PARAMETERS USED',
                    style={'textAlign': 'left'}),
            html.P(f"Required accuracy: eps = {self.__parameter_eps}",
                   style={'color': '#212121'}),
            html.P(f"Reliability parameter: r = {self.__parameter_r}",
                   style={'color': '#212121'}),
            html.P(f"Limiting the number of iterations: iters_limit = {self.__parameter_iters_limit}",
                   style={'color': '#212121'}),
            ],
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", }
        )

    def __render_solution_description(self):
        return html.Div(children=[
            html.H2('Solution',
                    style={'textAlign': 'left'}),
            html.P(
                f"Total trials number: {self.__trial_number} (global trials number - {self.__global_iter_number}, local trials number - {self.__local_iter_number})",
                style={'color': '#212121'}),
            html.P(f"{round(self.__best_value, 6)}", style={'font-size': '2.0em', 'color': 'black'}),
            html.P(f"Best point: {self.__best_float_point_dictionary}, {self.__best_discrete_point_dictionary}",
                   style={'color': '#212121'}),
            html.P(f'Best trial number: {self.__best_trial_number}', style={'color': '#212121'}),
            html.P(f"Accuracy: {round(self.__accuracy, 6)}", style={'color': '#212121'}),
            html.P(f"*[c] - continuous variables, [d] - discrete variables",
                   style={'color': '#212121', 'font-size': '0.8em'}),
        ], className="best trial description",
            style={'width': '45%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def __render_optimization_time(self):
        return html.Div(children=[
            html.H2('Optimization Time',
                    style={'textAlign': 'left'}),

            html.P(f"Total optimization time: {round(self.__solve_time, 3)} sec.", style={'color': '#212121'}),

            html.Div(children=[
                dcc.Graph(
                    figure={
                        "data": [{
                            "x": self.__dfSDI['trial'],
                            "y": self.__optimization_time,
                            'type': 'lines',
                            'marker': {'color': self.__base_color}
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

    def __hide_IQR(self, hide=False):
        if hide:
            Q1 = self.__dfSDI['objective_func'].quantile(0.25)
            Q3 = self.__dfSDI['objective_func'].quantile(0.75)
            mid = self.__dfSDI['objective_func'].median()
            IQR = Q3 - Q1

            # Задание надёжных границ с использованием IQR
            up = mid + 1.5 * IQR

            # Отсеивание на основе заданных границ
            self.__dfSDI = self.__dfSDI[(self.__dfSDI['objective_func'] <= up)]
        else:
            self.__dfSDI = self.__dfSDI_original
    def __render_iteration_characteristic(self):
        return html.Div([
            html.H2('Objective Function Value Сhange',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure=((px.scatter(self.__dfSDI, x='trial', y='objective_func',
                                            color_discrete_sequence=[self.__base_color], marginal_y="histogram",
                                            trendline="expanding", trendline_options=dict(function="min"))).update_layout(legend=
                                {'orientation': "h", 'y': -0.25},
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

    def __scatter_matrix_figure(self):
        fig = px.scatter_matrix(self.__dfSDI, dimensions=self.parameters_names, color="objective_func", opacity=0.7,
                                color_continuous_scale=self.__choosen_color, width=len(self.parameters_names) * 240,
                                height=len(self.parameters_names) * 240)
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF')
        return fig

    def __render_parameters_dependence(self):
        return html.Div([
            html.H2('Parameters Dependence',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure=self.__scatter_matrix_figure(),
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '100%'}),
            ], style={'maxWidth': '100%', 'maxHeight': '600px', "overflow": "scroll"}),
        ], style={'width': '100%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", 'height': '750px'})

    def __render_parameters_importance(self):
        return html.Div([
            html.H2('Parameters Importance',
                    style={'textAlign': 'left'}),
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': self.parameters_names,
                             'y': self.__importance,
                             'orientation': 'v',
                             'type': 'bar',
                             'text': self.__importance,
                             'textposition': 'outside',
                             'textfont' : "black",
                             'text_auto':'.2s',
                             'marker': {
                                 'color': self.__base_color,
                                 'line': {'color': self.__base_color, 'width': '1'}
                             }
                             },
                        ],
                        'layout': {
                            'xaxis': {
                                #'range': [0, 1],
                                'anchor': 'y',
                                'tickfont': {'size': '10'},
                                'tickangle':-90
                            },
                            'yaxis': {
                                'range': [0, 1],
                                'anchor': 'x',
                                'title': {'text': 'importance value'},
                                'tickfont': {'size': '10'}
                            },
                            'title': "Importance by contribution</br></br>to the scatter of the objective_func",
                            'paper_bgcolor': '#FFFFFF',
                            'plot_bgcolor': '#FFFFFF',
                            'margin':dict(t=130, b=200, r=100)
                        }
                    },
                    config={
                        'scrollZoom': True,  # True, False
                        'showTips': True,  # True, False
                        'displayModeBar': True,  # True, False, 'hover'
                    },
                )
            ])
        ], style={'width':'40%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF"})

    def __render_multidimensional_representation(self):
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
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF"})

    def __render_surface_and_level_lines(self):
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
                        self.__float_parameters_names,
                        self.__float_parameters_names[0],
                        id='crossfilter-xaxis-column-1',
                    ),
                    html.Label('Y-axis parameter'),
                    dcc.Dropdown(
                        self.__float_parameters_names,
                        self.__float_parameters_names[1],
                        id='crossfilter-yaxis-column-1',
                    ),
                    html.Label('Show subplots'),
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
                ], style={'width': '75%'}),
            ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '600px'}),
        ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def __render_objective_function_values_scatter(self):
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
                                self.__float_parameters_names,
                                self.__float_parameters_names[0],
                                id='crossfilter-xaxis-column', style={'width': '60%'}
                            ),
                            dcc.Graph(
                                id='continuous_scatter_figure',
                                config={
                                    'displayModeBar': True,  # True, False, 'hover'
                                },
                            )
                        ], style={'width': '100%'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '70%'}),
                ]),
                dcc.Tab(label='Discrete Parameters', children=[
                    html.Div([
                        html.Div(children=[
                            html.Br(),
                            html.Label('X-axis parameter'),
                            dcc.Dropdown(
                                self.__discrete_parameters_names,
                                self.__discrete_parameters_names[0],
                                id='crossfilter-xaxis-column-2', style={'width': '60%'}
                            ),
                            dcc.Graph(
                                id='discrete_scatter_figure',
                                config={
                                    'displayModeBar': True,  # True, False, 'hover'
                                },
                            )
                        ], style={'width': '100%'}),
                    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '90%', 'height': '70%'}),
                ])
            ], style={'width': '90%'})
        ], style={'width': '60%', "background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def __render_archive(self, df):
        return html.Div(children=[
            html.P(f"*[c] - continuous variables, [d] - discrete variables",
                   style={'text-align': 'right', 'color': '#212121', 'font-size': '0.8em'}),
            html.H1('All Trials Archive',
                    style={'textAlign': 'left'}),
            create_dash_table_from_dataframe(df),
            html.Div(id='archive_figure', style={'width': '95%'})
        ], style = {"background-color": "#FFFFFF", "border": "20px solid #FFFFFF", })

    def __render_page_content(self, pathname, color_theme, hide):
        self.__hide_IQR(hide)
        self.__choosen_color = self.__color_themes_dict[color_theme]
        self.__base_color = self.__choosen_color[0]
        if pathname == "/":
            return [
                html.Div([
                    self.__render_parameters_description(),
                    self.__render_problem_description(),
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                html.Div([
                    self.__render_solution_description(),
                    self.__render_optimization_time()
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                self.__render_iteration_characteristic(),
            ]

        elif pathname == "/archive":
            if self.__mode == 'Release':
                return [
                    self.__render_archive(self.__dfSDI_original[['trial'] + self.parameters_names + ["objective_func"]])
                ]
            elif self.__mode == 'Debug':
                return [
                    self.__render_archive(self.__dfSDI_original)
                ]
        elif pathname == "/analytics":
            return [
                html.Div(children=[
                    html.P(f"*[c] - continuous variables, [d] - discrete variables",
                       style={'text-align': 'right', 'color': '#212121', 'font-size': '0.8em'}),
                ], style={"background-color": "#FFFFFF", "border": "20px solid #FFFFFF"}),
                self.__render_multidimensional_representation(),
                self.__render_surface_and_level_lines(),
                html.Div([
                    self.__render_objective_function_values_scatter(),
                    self.__render_parameters_importance()
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                html.Div([
                    self.__render_parameters_dependence(),
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

    def __update_discrete_scatter_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.__discrete_parameters_names[0]

        fig = px.violin(self.__dfSDI,
                        x=xaxis_column_name, y='objective_func',
                        title="Scatter of objective function values</br></br>by selected discrete parameter",
                        color_discrete_sequence=[self.__base_color]
                        )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective_func')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False)
        return fig

    def __update_continuous_scatter_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.__float_parameters_names[0]

        fig = px.scatter(self.__dfSDI,
                         x=xaxis_column_name,
                         y='objective_func',
                         color=self.__dfSDI['trial'][::-1],
                         color_continuous_scale=list(reversed(self.__choosen_color)),
                         title="Scatter of objective function values</br></br>by selected continuous parameter",
                         opacity=0.3
                         )

        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective_func')
        fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF', showlegend=False,
                          coloraxis_colorbar=dict(title="trial number"))
        return fig

    def __calculate_data(self, xaxis_column_name, yaxis_column_name, calc):
        if xaxis_column_name == None:
            xaxis_column_name = self.__float_parameters_names[0]
        if yaxis_column_name == None:
            xaxis_column_name = self.__float_parameters_names[1]

        df = None

        # берем окрестность лучшего сочетания дискретных параметров
        for param in self.__discrete_parameters_names:
            df = self.__dfSDI.loc[self.__dfSDI[param] == self.__best_discrete_point_dictionary[param]]

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

        bounds_x = list(self.__float_parameters_bounds[(self.__float_parameters_names).index(xaxis_column_name)].values())[0]
        bounds_y = list(self.__float_parameters_bounds[(self.__float_parameters_names).index(yaxis_column_name)].values())[0]

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

    def __surface_figure(self, xaxis_column_name, yaxis_column_name, calc, showBars, xi, yi, Z, x, y, z):
        if calc == 'interpolation' or calc == 'approximation':
            surface = go.Surface(x=xi, y=yi, z=Z, colorscale=self.__choosen_color, opacity=1, colorbar=dict(title="objective_func"))
            fig = go.Figure(data=[surface])
            if showBars:
                fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        elif calc == 'none (by trials points)':
            surface = go.Mesh3d(x=x, y=y, z=z, showscale=True, intensity=z, colorscale=self.__choosen_color, opacity=1)
            fig = go.Figure(data=[surface])

        fig.add_scatter3d(x=x, y=y, z=z, mode='markers', name='trials points',
                          marker=dict(size=2, color='red', opacity=0.7))
        fig.add_scatter3d(x=[self.__best_float_point_dictionary[xaxis_column_name]],
                          y=[self.__best_float_point_dictionary[yaxis_column_name]],
                          z=[self.__best_value], name='best trial point',
                          mode='markers', marker=dict(size=3, color='green', opacity=1))
        fig.update_layout(title='Surface in cross-section of the '+ str(self.__best_discrete_point_dictionary), scene=dict(xaxis_title=xaxis_column_name, yaxis_title=yaxis_column_name,
                          zaxis_title='objective_func'), paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#FFFFFF', showlegend=False, height=590,
                          template="none")
        return fig

    def __lines_level_figure(self, xaxis_column_name, yaxis_column_name, calc, showBars, bounds_x, bounds_y, xi, yi, Z, x, y, z):
        if calc == 'interpolation' or calc == 'approximation':
            fig = go.Figure(data=[go.Contour(x=xi, y=yi, z=Z, colorscale=self.__choosen_color,
                                             colorbar=dict(title='objective_func'))])
        elif calc == 'none (by trials points)':
            fig = go.Figure(data=[go.Contour(x=x, y=y, z=z, colorscale=self.__choosen_color,
                                             colorbar=dict(title='objective_func', titleside='right'))])

        fig.add_scatter(x=x, y=y, mode='markers', name='trials points', marker=dict(size=2, color='red', opacity=0.7))
        fig.add_scatter(x=[self.__best_float_point_dictionary[xaxis_column_name]],
                        y=[self.__best_float_point_dictionary[yaxis_column_name]],
                        mode='markers', name='best trial point', marker=dict(size=4, color='green', opacity=1))

        fig.update_layout(title='Heatmap in cross-section of the '+ str(self.__best_discrete_point_dictionary),
                          paper_bgcolor='#FFFFFF',
                          plot_bgcolor='#FFFFFF', showlegend=True, height=590,
                          legend={'orientation': "h"},
                          xaxis_range=[bounds_x[0], bounds_x[1]], yaxis_range=[bounds_y[0], bounds_y[1]],
                          xaxis_title=xaxis_column_name,
                          yaxis_title=yaxis_column_name)
        if showBars:
            fig.add_trace(go.Histogram(
                y=y,
                xaxis='x2',
                marker=dict(
                    color=self.__base_color
                ),
                name=yaxis_column_name + ' values histogram'
            ))
            fig.add_trace(go.Histogram(
                x=x,
                yaxis='y2',
                marker=dict(
                    color=self.__base_color
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

    def __update_surface_or_lines_level_figure(self, xaxis_column_name=None, yaxis_column_name=None, type='3D Surface', calc='interpolation', showBars=True):
        bounds_x, bounds_y, x, y, z, xi, yi, Z = self.__calculate_data(xaxis_column_name, yaxis_column_name, calc)
        if type == '3D Surface':
            return self.__surface_figure(xaxis_column_name, yaxis_column_name, calc, showBars, xi, yi, Z, x, y, z)
        elif type == 'Heatmap':
            return self.__lines_level_figure(xaxis_column_name, yaxis_column_name, calc, showBars, bounds_x, bounds_y, xi, yi, Z, x, y, z)

    def __update_multidimensional_figure(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.parameters_names
        xaxis_column_name = ['objective_func'] + xaxis_column_name

        xaxis_column_name_dict = {}
        for name in xaxis_column_name:
            replace = ''
            if name in self.__discrete_parameters_names:
                self.__dfSDI[name + '_cat'] = self.__dfSDI[name].astype('category').cat.codes
                replace = name+'_cat'
            else:
                replace = name
            xaxis_column_name_dict[replace] = name


        fig = px.parallel_coordinates(self.__dfSDI, color="objective_func",
                                      dimensions=xaxis_column_name_dict.keys(),
                                      labels=xaxis_column_name_dict,
                                      color_continuous_scale=self.__choosen_color)

        fig.update_layout(
            xaxis=dict(
                title='parameters',
                ticktext=xaxis_column_name
            ),
            yaxis=dict(
                title='objective_func'
            ),
            paper_bgcolor = '#FFFFFF', plot_bgcolor = '#FFFFFF'
        )

        fig.update_traces(unselected_line_opacity=0.5, selector=dict(type='parcoords'))

        return fig

    def __read_data_from_json(self, contents):
        if self.__init and contents is not None:
            content_type, content_string = contents[0].split(',')
            decoded = base64.b64decode(content_string)
            self.__data = json.loads(decoded)
            self.launch()
        else:
            self.__init = True
        return html.Div(id='hidden-div', style={'display':'none'})

    def __update_archive_figure(self, rows, derived_virtual_selected_rows):
        if derived_virtual_selected_rows is None:
            derived_virtual_selected_rows = []


        dff = self.__dfSDI_original if rows is None else pd.DataFrame(rows)

        ids = []
        check = list(~dff['trial'].isin(self.__dfSDI['trial']))
        for i in range(len(dff)):
           if check[i]:
               ids.append(i)

        colors = ['#31B37C' if i in derived_virtual_selected_rows else
                  "red" if i in ids else
                  self.__base_color
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
                            "title": {"text": "objective_func"}
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