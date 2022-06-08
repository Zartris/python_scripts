from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import colorhash
from matplotlib.patches import Patch
from calendar import monthrange

# create a column with the color for each department
from pandas import Timestamp


def color(row):
    c_dict = {'MKT': '#E64646', 'FIN': '#E69646', 'ENG': '#34D05C', 'PROD': '#34D0C3', 'IT': '#3475D0'}
    return colorhash.ColorHash(row.TaskDescription).hex


def get_number_of_days_in_month(year, month):
    return monthrange(year, month)[1]


def get_xticks_from_data(df):
    xticks = [0]  # start date added
    start_of_project = df.Start.min()
    c_year = start_of_project.year
    c_month = start_of_project.month
    s_day = start_of_project.day
    days = get_number_of_days_in_month(c_year, c_month)
    xticks.append(days + 1 - s_day)
    end_of_project = df.End.max()
    while not (c_month == end_of_project.month and c_year == end_of_project.year):
        # update month and year
        if c_month == 12:
            c_year += 1
            c_month = 1
        else:
            c_month += 1
        xticks.append(xticks[-1] + get_number_of_days_in_month(c_year, c_month))
    xticks = np.array(xticks)
    xticks[-1] = xticks[-2] + (end_of_project.day - 1)
    return xticks


if __name__ == '__main__':
    title = 'Timeline for visit abroad'
    bg_color = '#36454F'
    text_color = 'w'
    show_minor = False
    dpi = 400
    y_label_rotation = 0
    x_label_rotation = 45
    margin_left = 0.13
    margin_right = 0.02
    margin_top = 0
    margin_bottom = 0.1
    make_legends = False
    data_path = Path("data\\gantt_timeline_abroad.xlsx")
    save = True

    ##### DATA LOAD #####
    df = pd.read_excel(str(data_path))
    # Remove empty rows or invalid rows
    df.dropna(inplace=True)
    # Sort the data by start and end date:
    df.sort_values(by=['Start', 'End'], ascending=False, inplace=True)
    # project start date
    proj_start = df.Start.min()

    # number of days from project start to task start
    df['start_num'] = (df.Start - proj_start).dt.days

    # number of days from project start to end of tasks
    df['end_num'] = (df.End - proj_start).dt.days

    # days between start and end of each task
    df['days_start_to_end'] = df.end_num - df.start_num
    df.Completion = 1

    # days between start and current progression of each task
    df['current_num'] = (df.days_start_to_end * df.Completion)

    # days between start and current progression of each task
    # df['current_num'] = (df.days_start_to_end * df.Completion)

    df['color'] = df.apply(color, axis=1)

    ##### PLOT #####
    if make_legends:
        fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [12, 1]}, facecolor=bg_color)
        ax1.set_facecolor(bg_color)
        # ##### LEGENDS #####
        legend_elements = [Patch(facecolor='#E64646', label='Marketing'),
                           Patch(facecolor='#E69646', label='Finance'),
                           Patch(facecolor='#34D05C', label='Engineering'),
                           Patch(facecolor='#34D0C3', label='Production'),
                           Patch(facecolor='#3475D0', label='IT')]

        legend = ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)
        plt.setp(legend.get_texts(), color=text_color)

        # clean second axis (legend)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
    else:
        fig, ax = plt.subplots(figsize=(20, 8), facecolor=bg_color)  # create figure and axes
    fig.set_dpi(dpi)  # 400 is good quality
    plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1 - margin_right, top=1 - margin_top,
                        wspace=0, hspace=0)
    ax.set_facecolor(bg_color)

    # bars
    ax.barh(df.TaskDescription, df.current_num, left=df.start_num, color=df.color)
    ax.barh(df.TaskDescription, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.5)
    # plt.show()
    # test = pd.date_range(proj_start, end=df.End.max()).strftime("%m-%y")

    # for idx, row in df.iterrows():
    #     ax.text(row.end_num + 0.1, idx, f"{int(row.Completion * 100)}%", va='center', alpha=0.8, color='w')
    #     ax.text(row.start_num - 0.1, idx, row.Task, va='center', ha='right', alpha=0.8, color='w')

    # grid lines
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

    # Compute label days shown:
    end_of_project = df.End.max()
    end_label = pd.Timestamp(year=end_of_project.year, month=end_of_project.month,
                             day=get_number_of_days_in_month(end_of_project.year, end_of_project.month), hour=0)
    # ticks
    xticks = get_xticks_from_data(df)
    xticks_labels = pd.date_range(proj_start, end=end_label).strftime("%d-%b-%y")
    xticks_minor = np.arange(0, df.end_num.max() + 1, 1)  # Per day minor
    ax.set_xticks(xticks)
    if show_minor:
        ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(ax.get_xticks(), rotation=x_label_rotation)
    ax.set_xticklabels(xticks_labels[xticks], color=text_color)
    ax.set_yticks(df.TaskDescription)
    ax.tick_params(axis='y', colors=text_color)
    ax.tick_params(axis='y', labelrotation=y_label_rotation)

    plt.setp([ax.get_xticklines()], color=text_color)

    # align x axis
    ax.set_xlim(0, df.end_num.max() + 1)

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('w')

    plt.suptitle(title, color=text_color)


    if save:
        plt.savefig(str(Path(data_path.parent, 'gantt.png')))
    else:
        plt.show()
