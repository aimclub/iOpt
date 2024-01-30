import dash
from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
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
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "font-size": "1.1em",
}

def generate_table(dataframe):
    return html.Div([dash_table.DataTable(dataframe.to_dict('records'), [{"name": i, "id": i} for i in dataframe.columns],
        filter_action="native", id='datatable-interactivity', sort_action="native",
        row_selectable="multi", selected_rows=[],
        sort_mode="multi", page_size=15)],
        style={'maxWidth': '95%', 'maxHeight': '800px', "overflow": "scroll"})

class StaticDashboard():
    def __init__(self, data_path):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
        self.app.config['suppress_callback_exceptions'] = True

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
        self.normalizate_values_T_dictionary = None
        self.value_colors_dictionary = None
        self.values_names_T = None
        self.normalizate_values = None

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
        self.task_name = "example" # TBD
        self.dfSDI = pd.DataFrame(data['SearchDataItem'])
        dfFV = pd.DataFrame(data['float_variables'])
        dfDV = pd.DataFrame(data['discrete_variables'])
        dfBT = pd.DataFrame(data['best_trials'])
        dfS = pd.DataFrame(data['solution'])

        self.float_parameters_bounds = data['float_variables']
        self.discrete_parameters_valid_values = data['discrete_variables']

        self.float_parameters_names = [f"{x} [fv]" for x in [*dfFV]]
        self.discrete_parameters_names = [f"{x} [dv]" for x in [*dfDV]]

        self.parameters_names = self.float_parameters_names + self.discrete_parameters_names

        self.best_value = dfBT['__z'][0]
        self.accuracy = float(dfS['solution_accuracy'])
        self.trial_number = int(dfS['number_of_trials'])
        self.global_iter_number = int(dfS['number_of_global_trials'])
        self.local_iter_number = int(dfS['number_of_local_trials'])
        self.solve_time = float(dfS['solving_time'])

        FVs = pd.DataFrame(self.dfSDI['float_variables'].to_list(), columns=self.float_parameters_names,
                           index=self.dfSDI.index)
        DVs = pd.DataFrame(self.dfSDI['discrete_variables'].to_list(), columns=self.discrete_parameters_names,
                           index=self.dfSDI.index)

        self.dfSDI = self.dfSDI[["__z", "x", "delta", "globalR", "localR"]]
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

        self.best_trial_number = 0
        for elem in self.dfSDI['__z']:
            self.best_trial_number += 1
            if elem == self.best_value:
                break

        self.best_float_point_dictionary = dict(
            zip(self.float_parameters_names, [round(elem, accuracy_in_signs) for elem in dfBT['float_variables'][0]]))
        self.best_discrete_point_dictionary = dict(zip(self.discrete_parameters_names, dfBT['discrete_variables'][0]))

        self.optimization_time = list(np.linspace(0, self.solve_time, self.trial_number))

        self.trial_names = ['trial ' + str(index + 1) for index in range(self.trial_number)]

    def normalizateData(self):
        minZ = min(self.dfSDI['__z'])
        maxZ = max(self.dfSDI['__z'])

        self.normalizate_values = []
        values_names = []
        self.normalizate_values.append([float(elem - minZ) / (maxZ - minZ) for elem in self.dfSDI['__z']])
        values_names.append(['__z' for _ in range(self.trial_number)])

        for i in range(len(self.float_parameters_names)):
            lb = list(self.float_parameters_bounds[i].values())[0][0]
            rb = list(self.float_parameters_bounds[i].values())[0][1]
            self.normalizate_values.append(
                [float(elem - lb) / (rb - lb) for elem in list(self.dfSDI[self.float_parameters_names[i]].values)])
            values_names.append([self.float_parameters_names[i] for _ in range(self.trial_number)])

        for i in range(len(self.discrete_parameters_names)):
            vals = list(self.discrete_parameters_valid_values[i].values())[0]
            self.normalizate_values.append([float(vals.index(elem)) / len(vals) for elem in
                                            list(self.dfSDI[self.discrete_parameters_names[i]].values)])
            values_names.append([self.discrete_parameters_names[i] for _ in range(self.trial_number)])

        normalizate_values_T = list(map(list, zip(*self.normalizate_values)))
        self.normalizate_values_T_dictionary = dict(zip(self.trial_names, normalizate_values_T))
        self.values_names_T = list(map(list, zip(*values_names)))

    def calculateColors(self):
        value_colors = []
        current_trial = 1
        for elem in self.normalizate_values[0]:
            if current_trial == self.best_trial_number:
                value_colors.append("rgba(" + str(0) + "," + str(0) + "," + str(0) + ',' + str(1) + ")")
            else:
                color = int(10 + (255 - 10) * elem)
                value_colors.append(
                    "rgba(" + str(color) + "," + str(color) + "," + str(color) + ',' + str(0.3 * (1 - elem)) + ")")
            current_trial += 1
        self.value_colors_dictionary = dict(zip(self.trial_names, value_colors))

    def calculateLowerEnvelopeValues(self):
        self.values_update_best = []
        self.points_update_best = []
        min_value = max(self.dfSDI['__z']) + 1
        for value, point in zip(self.dfSDI['__z'], self.dfSDI['trial']):
            if min_value > value:
                min_value = value
                self.values_update_best.append(value)
                self.points_update_best.append(point)

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
            self.importance.append(parameter_importance_value / (len(uniq_parameter_values) * (maxZ - minZ)))

    def createSidebar(self):
        self.sidebar = html.Div(
            [
                dbc.Nav(
                    [
                        html.Img(src=r'assets/iOptdash_light_.png', alt='image', width="224px"),
                        html.Hr(),
                        dbc.NavLink("Home", href="/", active="exact", style={"color": "#AAAAAA"}),
                        dbc.NavLink("Analytics", href="/page-1", active="exact", style={"color": "#AAAAAA"}),
                        dbc.NavLink("Archive", href="/page-2", active="exact", style={"color": "#AAAAAA"}),
                        html.Hr(style={'width': '8rem'}),
                        html.Div([dbc.NavLink("Github Source", href="https://github.com/aimclub/iOpt",
                                              style={"color": "#808080"})])

                    ],
                    vertical=True,
                    pills=True,
                ),
            ], className="sidebar discription",
            style=SIDEBAR_STYLE,
        )

    def createContentPage(self):
        self.content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

    def launch(self):
        self.readData(self.data)
        del self.data
        self.data = None
        self.normalizateData()
        self.calculateColors()
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
            html.P(
                f"Problem name: {self.task_name}"),
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
            style={'width': '43%', "background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", }
        )

    def render_solution_discription(self):
        return html.Div(children=[
            html.H2('Best Trial',
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
            style={'width': '43%', "background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", })

    def render_optimization_time_plot(self):
        return html.Div(children=[
            html.H2('Optimization Time',
                    style={'textAlign': 'left'}),

            html.P(f"Total optimization time: {round(self.solve_time, 3)} sec.", style={'color': '#212121'}),

            dcc.Graph(
                figure={
                    "data": [{
                        "x": self.optimization_time,
                        "y": self.dfSDI['trial'],
                        "type": "lines",
                        'marker': {'color': '#0D0B93'}
                    }, ],
                    "layout": {
                        'paper_bgcolor': '#F7F7F7',
                        'plot_bgcolor': '#F7F7F7',
                        'xaxis': {'anchor': 'y', 'title': {'text': 'trial end time, sec.'}},
                        'yaxis': {'anchor': 'x', 'title': {'text': 'trial number'}}
                    },
                },
                config={
                    'displayModeBar': True,  # True, False, 'hover'
                },
            )
        ], style={'width': '55%', "background-color": "#F7F7F7",
                  "border": "3px #F7F7F7 outset", })  # "border":"1px black solid",

    def render_iteration_characteristic(self):
        return html.Div([
            html.H2('Objective Function Value Сhange',
                    style={'textAlign': 'left'}),
            html.Div(children=[
                html.Div(children=[
                    dcc.Graph(
                        figure={
                            "data": [{
                                "x": self.points_update_best,
                                "y": self.values_update_best,
                                "mode": "lines",
                                'marker': {'color': 'rgba(13, 11, 147, 0.6)'},
                                'name': 'Current best value of objective function value',
                            }, {
                                "x": self.dfSDI['trial'],
                                "y": self.dfSDI['__z'],
                                'type': 'scatter',
                                "mode": "markers",
                                'name': 'Objective function value',
                                'marker': {'color': '#0D0B93'}
                            }, ],
                            "layout": {
                                'legend': {'orientation': "h", 'y': -0.25},
                                'xaxis': {'anchor': 'x', 'title': {'text': 'trial number'}},
                                'yaxis': {'anchor': 'y', 'title': {'text': 'objective function value'}},
                                'paper_bgcolor': '#F7F7F7',
                                'plot_bgcolor': '#F7F7F7',
                            },
                        },
                        config={
                            'displayModeBar': True,  # True, False, 'hover'
                        },
                    )
                ], style={'width': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], style={"background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", })

    def render_multidimentional_visualization(self):
        return html.Div(children=[
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
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': '90%',
                  "background-color": "#F7F7F7", "border": "3px #F7F7F7 outset"})

    def render_3D_graph(self):
        return html.Div(children=[
            html.H2('Visualization in cross-section of the best point using discrete parameters',
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
        ], style={"background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", })

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
        ], style={"background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", })

    def render_importance(self):
        return html.Div([
            html.H2('Parameters Importance',
                    style={'textAlign': 'left'}),
            html.Div([
                html.Div(children=[
                    html.Div(children=[
                        dcc.Graph(
                            figure={
                                'data': [
                                    {'x': self.importance,
                                     'y': self.parameters_names,
                                     'orientation': 'h',
                                     'type': 'bar',
                                     'text': self.importance,
                                     'textposition': 'outside',
                                     'marker': {
                                         'color': 'rgba(13, 11, 147, 0.8)',
                                         'line': {'color': 'rgba(13, 11, 147, 1.0)', 'width': '1'}
                                     }
                                     },
                                ],
                                'layout': {
                                    'xaxis': {
                                        'range': [0, 1],
                                        'anchor': 'y',
                                        'title': {'text': 'trial number'},
                                        'tickfont': {'size': '12'},
                                    },
                                    'yaxis': {
                                        'anchor': 'x',
                                        'title': {'text': 'parameter'},
                                        'tickfont': {'size': '12'}
                                    },
                                    'title': "Importance of the hyperparameter",
                                    'paper_bgcolor': '#F7F7F7',
                                    'plot_bgcolor': '#F7F7F7',
                                }
                            },
                            config={
                                'scrollZoom': False,  # True, False
                                'showTips': False,  # True, False
                                'displayModeBar': False,  # True, False, 'hover'
                            },
                        )
                    ], style={'width': '50%'}),
                    html.Div(children=[
                        dcc.Graph(
                            figure={
                                "data": [{
                                    "x": [x for item in (self.float_parameters_names) for x in self.dfSDI[item]],
                                    "y": [item for item in (self.float_parameters_names) for _ in
                                          range(len(self.dfSDI['__z']))],
                                    'type': 'scatter',
                                    "mode": "markers",
                                    'name': "",
                                    'marker': {
                                        'color': [x for x in self.dfSDI['__z'] for _ in
                                                  range(len(self.float_parameters_names))],
                                        'colorscale': "Viridis",
                                        'showscale': True,
                                        'opacity': 0.8
                                    }
                                },
                                    {
                                        "x": [float(
                                            list(self.discrete_parameters_valid_values[i].values())[0].index(x)) / len(
                                            list(self.discrete_parameters_valid_values[i].values())[0]) for i in
                                              range(len(self.discrete_parameters_names)) for x in
                                              self.dfSDI[self.discrete_parameters_names[i]]],
                                        "y": [item for item in (self.discrete_parameters_names) for _ in
                                              range(len(self.dfSDI['__z']))],
                                        'text': [x for item in (self.discrete_parameters_names) for x in
                                                 self.dfSDI[item]],
                                        'type': 'scatter',
                                        "mode": "markers",
                                        'name': "",
                                        'marker': {
                                            'color': [x for x in self.dfSDI['__z'] for _ in
                                                      range(len(self.discrete_parameters_names))],
                                            'colorscale': "Viridis",
                                            'showscale': True,
                                            'opacity': 0.8
                                        }
                                    }],
                                "layout": {
                                    'xaxis': {'anchor': 'x', 'title': {'text': 'objective function value'}},
                                    'yaxis': {'anchor': 'y', 'showticklabels': False},
                                    'paper_bgcolor': '#F7F7F7',
                                    'plot_bgcolor': '#F7F7F7',
                                    'title': 'Scatter of objective function values by one parameter',
                                    'showlegend': False
                                },
                            },
                            config={
                                'displayModeBar': True,  # True, False, 'hover'
                            },
                        )
                    ], style={'width': '50%'}),
                ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row'}),
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ], style={"background-color": "#F7F7F7", "border": "3px #F7F7F7 outset", })

    def render_page_content(self, pathname):
        if pathname == "/":
            return [
                html.Div([
                    self.render_task_discription(),
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                html.Div([
                    self.render_solution_discription(),
                    html.Div(children=[], style={'width': '2%'}),
                    self.render_optimization_time_plot()
                ], style={'display': 'flex', 'flexDirection': 'row'}),

                html.Div(children=[
                    html.Br()
                ], style={'width': '20px'}),

                self.render_iteration_characteristic(),
                self.render_multidimentional_visualization(),

                html.Div(children=[
                    html.Br()
                ], style={'width': '20px'}),
            ]

        elif pathname == "/page-2":
            return [
                html.H1('All Trials Archive',
                        style={'textAlign': 'left'}),
                generate_table(self.dfSDI),
                html.Div(id='datatable-interactivity-container')

            ]
        elif pathname == "/page-1":
            return [
                self.render_3D_graph(),
                self.render_objective_function_value_scatter(),
                self.render_importance()
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

        fig = px.scatter(self.dfSDI,
                         x=xaxis_column_name, y='__z',
                         color=self.dfSDI['trial'][::-1],
                         color_continuous_scale="Viridis",
                         title="Scatter of objective function values by one parameter"
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function value')
        fig.update_layout(paper_bgcolor='#F7F7F7', plot_bgcolor='#F7F7F7', showlegend=False)
        return fig

    def update_graph_2(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]

        fig = px.scatter(self.dfSDI,
                         x=xaxis_column_name,
                         y='__z',
                         color=self.dfSDI['trial'][::-1],
                         color_continuous_scale="Viridis",
                         title="Scatter of objective function values by one parameter"
                         )
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title='objective function value')
        fig.update_layout(paper_bgcolor='#F7F7F7', plot_bgcolor='#F7F7F7', showlegend=False)
        return fig

    def update_graph_3(self, xaxis_column_name=None, yaxis_column_name=None, type='3D Surface', calc='interpolation'):
        if xaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[0]
        if yaxis_column_name == None:
            xaxis_column_name = self.float_parameters_names[1]

        df = self.dfSDI

        # берем окрестность лучшего сочетания дискретных параметров
        for param in self.discrete_parameters_names:
            df = df.loc[df[param] == self.best_discrete_point_dictionary[param]]

        # использовать окрестность лучшего решения
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
                surface = go.Surface(x=xi, y=yi, z=Z, colorscale="Viridis", opacity=1)
                fig = go.Figure(data=[surface])
                fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                                  highlightcolor="limegreen", project_z=True))
            elif calc == 'none (by trials points)':
                surface = go.Mesh3d(x=x, y=y, z=z, showscale=True, intensity=z, colorscale="Viridis", opacity=1)
                fig = go.Figure(data=[surface])

            fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red', opacity=0.4))
            fig.update_layout(title='3D Surface in cross-section of the best point', paper_bgcolor='#F7F7F7',
                              plot_bgcolor='#F7F7F7', showlegend=False, height=700,
                              template="none")
        else:
            if calc == 'interpolation' or calc == 'approximation':
                fig = go.Figure(data=[go.Contour(x=xi, y=yi, z=Z, colorscale="Viridis")])
            elif calc == 'none (by trials points)':
                fig = go.Figure(data=[go.Contour(x=x, y=y, z=z, colorscale="Viridis")])  # , line_smoothing=0.85

            fig.update_layout(title='Level lines in cross-section of the best solution', paper_bgcolor='#F7F7F7',
                              plot_bgcolor='#F7F7F7', showlegend=False, height=700,
                              template="none")

        return fig

    def update_graph_4(self, xaxis_column_name=None):
        if xaxis_column_name == None:
            xaxis_column_name = self.parameters_names

        dffff = pd.DataFrame(data=self.normalizate_values_T_dictionary)
        for_del = list(set(self.parameters_names) - set(xaxis_column_name))
        indxs = [(self.parameters_names).index(el) for el in for_del]
        if len(indxs) != 0:
            dffff = dffff.drop(indxs)
        ll = self.values_names_T[0].copy()
        for i in sorted(indxs, reverse=True):
            del ll[i]
        fig = px.line(dffff, x=ll, y=list(dffff.columns.values),
                      title="Scatter of objective function values by one parameter",
                      markers=True, color_discrete_map=self.value_colors_dictionary)
        fig.update_xaxes(title='parameters')
        fig.update_yaxes(title='objective function value')
        fig.update_layout(paper_bgcolor='#F7F7F7', plot_bgcolor='#F7F7F7', showlegend=False)

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
        colors = ['#31B37C' if i in derived_virtual_selected_rows else '#0D0B93'
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
                            "title": {"text": "objective function value"}
                        },
                        "height": 250,
                        "weight": "95%",
                        "margin": {"t": 10, "l": 10, "r": 10},
                    },
                },
            )
        ]