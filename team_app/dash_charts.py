# Your answer - EX05 and EX06 (Role C). Write both functions in this one file.
import matplotlib.pyplot as plt
from dash_tools import week_table, HIGHLIGHT, GREY, PURPLE
import pandas as pd

def weekly_chart(df):
    """Line chart of requests per complete week, or None (see EX05)."""
    if df.empty:
      return None
    
    weak_tables = week_table(df)
    Final_result = weak_tables[weak_tables["complete"]]
    if Final_result.empty:
      return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Final_result.index, Final_result["requests"], color=PURPLE, marker="o")
    ax.set_ylim(bottom=0)                      
    ax.set_title("Requests per complete week")
    ax.set_ylabel("requests")
    fig.autofmt_xdate()                        
    plt.close(fig)

    return fig

def desk_chart(df):
    """Horizontal bars of median wait by desk with an action title, or None (see EX06)."""
    if df.empty:
      return None
    
    waits = df.groupby('desk')["wait_hours"].median().sort_values()
    counts = df["desk"].value_counts()

    top_desk = waits.index[-1]
    top_median  = waits.iloc[-1]

    labels = [f"{desk} (n={counts[desk]})" for desk in waits.index]
    colors = [HIGHLIGHT if desk == top_desk else GREY for desk in waits.index]

    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, waits.values, color=colors)
    ax.set_xlim(left=0)
    ax.set_xlabel("median wait (hours)")
    ax.set_title(f"{top_desk} has the longest waits: {top_median:.1f} hours median")
    plt.close(fig)

    return fig
