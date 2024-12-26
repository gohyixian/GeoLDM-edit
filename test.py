import gradio as gr
import plotly.graph_objects as go

# Sample dictionary with x-axis values as keys and y-axis frequencies as values
# data = {"A": 10, "B": 15, "C": 7, "D": 12}
data = \
    {3: 4, 4: 45, 5: 111, 6: 709, 7: 655, 8: 1043, 9: 1861, 10: 2884, 11: 3104, 12: 2910, 13: 2167, 14: 2439, 15: 2301, 16: 3117, 17: 2384, 18: 2611, 19: 3118, 20: 4890, 21: 4488, 22: 3356, 23: 4446, 24: 3814, 25: 4593, 26: 4770, 27: 5496, 28: 4772, 29: 3143, 30: 3582, 31: 3383, 32: 3312, 33: 2658, 34: 2162, 35: 1852, 36: 1357, 37: 927, 38: 919, 39: 739, 40: 674, 41: 607, 42: 487, 43: 360, 44: 673, 45: 184, 46: 212, 47: 105, 48: 186, 49: 73, 50: 38, 51: 20, 52: 42, 53: 3, 54: 93, 55: 35, 56: 12, 57: 9, 58: 18, 59: 5, 61: 9, 62: 1, 63: 5, 65: 2, 66: 20, 67: 28, 68: 1, 71: 1, 77: 1, 86: 1, 98: 1, 106: 2}

# Function to generate histogram from dictionary
def plot_histogram(data_dict):
    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())
    
    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
    fig.update_layout(
        title="Histogram",
        xaxis_title="X-Axis Values",
        yaxis_title="Frequencies"
    )
    return fig

# Gradio interface
demo = gr.Interface(
    fn=lambda: plot_histogram(data),
    inputs=[],
    outputs=gr.Plot(),
    live=False,
)

demo.launch()
