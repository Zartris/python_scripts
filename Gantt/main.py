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


if __name__ == '__main__':
    title = 'PROJECT XYZ'
    bg_color = '#36454F'
    text_color = 'w'

    data_path = Path("C:\\code\\python\\scripts\\Gantt\\data\\gantt_timeline_abroad.xlsx")

    ##### DATA LOAD #####
    df = pd.read_excel(str(data_path))

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
    fig, (ax, ax1) = plt.subplots(2, figsize=(16, 6), gridspec_kw={'height_ratios': [12, 1]}, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax1.set_facecolor(bg_color)
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
    xticks = [0]  # start date added
    start_of_project = df.Start.min()
    c_year = start_of_project.year
    c_month = start_of_project.month
    s_day = start_of_project.day
    days = get_number_of_days_in_month(c_year, c_month)
    xticks.append(days + 1 - s_day)
    end_of_project = df.End.max()
    end_label = pd.Timestamp(year=end_of_project.year, month=end_of_project.month,
                             day=get_number_of_days_in_month(end_of_project.year, end_of_project.month), hour=0)
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
    # ticks
    # xticks = np.arange(0, df.end_num.max() + 1, 30)
    xticks_labels = pd.date_range(proj_start, end=end_label).strftime("%m-%d")
    xticks_minor = np.arange(0, df.end_num.max() + 1, 1)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(ax.get_xticks(), rotation=30)
    labels = xticks_labels[xticks]
    ax.set_xticklabels(xticks_labels[xticks], color=text_color)
    ax.set_yticks([])

    plt.setp([ax.get_xticklines()], color=text_color)

    # align x axis
    ax.set_xlim(0, df.end_num.max())

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('w')

    plt.suptitle(title, color=text_color)

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
    plt.show()
    # plt.savefig('gantt.png', facecolor='#36454F')
