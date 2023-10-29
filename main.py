import pandas as pd
import taipy as tp
from taipy.gui import Gui

from taipy import Core
from pages import *

def create_chart(sm_results: pd.DataFrame, var: str):
    """Functions that create/update the chart table visible in the "Databases" page. This
    function is used in the "on_change" function to change the chart when the graph selected is changed.

    Args:
        sm_results (pd.DataFrame): the results database that comes from the state
        var (str): the string that has to be found in the columns that are going to be used to create the chart table

    Returns:
        pd.DataFrame: the chart with the proper columns
    """
    if var == 'Cost':
        columns = ['index'] + [col for col in sm_results.columns if var in col]
    else:
        columns = ['index'] + [col for col in sm_results.columns if var in col and 'Cost' not in col]

    chart = sm_results[columns]
    return chart


def on_change(state, var_name, var_value):
    """This function is called whener a change in the state variables is done. When a change is seen, operations can be created
    depending on the variable changed
    Args:
        state (State): the state object of Taipy
        var_name (str): the changed variable name
        var_value (obj): the changed variable value
    """
    # if the changed variable is the scenario selected
    if var_name == "selected_scenario" and var_value:
        if state.selected_scenario.results.is_ready_for_reading:
            # I update all the other useful variables
            update_variables(state)

    
    if var_name == 'sm_graph_selected' or var_name == "selected_scenario":
        # Update the chart table
        str_to_select_chart = None
        chart_mapping = {
            'Costs': 'Cost',
            'Purchases': 'Purchase',
            'Productions': 'Production',
            'Stocks': 'Stock',
            'Back Order': 'BO',
            'Product FPA': 'FPA',
            'Product FPB': 'FPB',
            'Product RP1': 'RP1',
            'Product RP2': 'RP2'
        }

        str_to_select_chart = chart_mapping.get(state.sm_graph_selected)
        state.chart = create_chart(state.sm_results, str_to_select_chart)

        # If we are on the 'Databases' page, we have to create a temp CSV file
        if state.page == 'Databases':
            state.d_chart_csv_path = PATH_TO_TABLE
            state.chart.to_csv(state.d_chart_csv_path, sep=',')


def on_init(state):
    state.state_id = str(os.urandom(32))
    update_scenario_selector(state)

pages = {
    "/": root_page,
    "data_plot": data_plot,
	"data_load": data_load,
	"data_browse": data_browse
}


if __name__ == "__main__":
    core = Core()
    core.run()
    # #############################################################################
    # PLACEHOLDER: Create and submit your scenario here                           #
    #                                                                             #
    # Example:                                                                    #
    # from configuration import scenario_config                                   #
    # scenario = tp.create_scenario(scenario_config)                              #
    # scenario.submit()                                                           #
    # Comment, remove or replace the previous lines with your own use case        #
    # #############################################################################

    gui = Gui(pages=pages)
    gui.run(title="FEA Dashboard")
