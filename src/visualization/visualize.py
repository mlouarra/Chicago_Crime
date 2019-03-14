
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models.widgets import Slider, CheckboxButtonGroup
from bokeh.layouts import row, WidgetBox

class Visu:

    """
    the aim of this class is to generate interactive chart plot
    """
    def __init__(self, config, df_merged):
        """

        :param config:
        :return:
        """
        self._config = config
        self._df_merged = df_merged

    def modify_doc(self, doc):
        """

        :return:
        """

        def my_color():
            """

            :return:
            """
            return self._config['Mycolor']['Color_1']

        def name_area():
            return self._config['NameArea']

        def name_crime():
            return self._config['NameCrime']

        def make_dataset(list_community_area, year):

            import random
            import copy

            df = self._df_merged[self._df_merged['year'] == year]
            df_grouped = df.groupby(['community_area_name', 'primary_type'],
                                    as_index=False).id.count().rename(columns={'id': 'nb_crimes'})

            list_df_total = []
            random.shuffle(my_color(), random.random)
            for i, community in enumerate(list_community_area):
                subset = df_grouped[df_grouped['community_area_name'] == community]
                subset_copy = copy.deepcopy(subset)
                subset_copy['nb_crimes'].fillna(0, inplace=True)
                subset_copy['color'] = my_color()[i]
                list_df_total.append(subset_copy)
            del df_grouped
            del subset
            return ColumnDataSource(pd.concat(list_df_total))


        def style(p):

            """

            :param p:
            :return:
            """
            p.y_range.start = 0
            p.x_range.range_padding = 0.05
            p.xgrid.grid_line_color = None
            p.xaxis.axis_label = "type of crime"
            p.xaxis.major_label_orientation = 1.2
            return p

        def make_plot(src):
            """

            :param src:
            :return:
            """
            # Blank plot with correct labels
            p = figure(plot_width=700, plot_height=900, title='crimes by community',
                       x_axis_label='community', y_axis_label='nb_crimes', x_range=list_primary_type)

            p.vbar(x='primary_type', top='nb_crimes', width=1, source=src,
                   color='color', hover_fill_color='color', line_color="white")
            # Hover tool with vline mode
            hover = HoverTool(tooltips=[('community_area_name', '@community_area_name'),
                                        ('primary_type', '@primary_type'),
                                        ('nb_crimes', '@nb_crimes')], mode='vline')
            p.add_tools(hover)
            # Styling
            p = style(p)
            return p

        def update(attr, old, new):
            """

            :param attr:
            :param old:
            :param new:
            :return:
            """
            carriers_to_plot = [carrier_selection.labels[i] for i in
                                carrier_selection.active]
            new_src = make_dataset(carriers_to_plot, year_select.value)
            src.data.update(new_src.data)

        list_primary_type = name_crime()
        list_community_area = name_area()
        carrier_selection = CheckboxButtonGroup(labels=list_community_area, active=[0])
        carrier_selection.on_change('active', update)
        year_select = Slider(start=2001, end=2017,
                             step=1, value=2001,
                             title='year')
        year_select.on_change('value', update)
        controls = WidgetBox(year_select, carrier_selection)
        initial_carriers = [carrier_selection.labels[i] for i in carrier_selection.active]
        src = make_dataset(initial_carriers, year_select.value)
        p = make_plot(src)
        layout = row(controls, p)
        doc.add_root(layout)