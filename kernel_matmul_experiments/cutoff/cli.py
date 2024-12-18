import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.integrate
import scipy.optimize
import scipy.special
from plotly.subplots import make_subplots
from tqdm import tqdm


def main():
    lengthscale = 1.0
    frequencies = np.linspace(0, 1.5, 500)
    epsilons = [1e-1, 1e-3, 1e-5]

    results = np.zeros((len(epsilons), len(frequencies), 2))

    for i, epsilon in enumerate(epsilons):
        for j, frequency in enumerate(tqdm(frequencies, desc=f"ε = {epsilon}")):

            def integrand(tau: np.ndarray) -> np.ndarray:
                return np.abs(
                    np.exp(-0.5 * tau**2 / lengthscale**2) * np.cos(2 * np.pi * frequency * tau)
                )

            complete_integral = scipy.integrate.quad(integrand, -np.inf, np.inf, limit=10000)[0]
            cutoff = np.sqrt(2) * lengthscale * scipy.special.erfinv(1 - epsilon)
            cutoff_integral = scipy.integrate.quad(integrand, -cutoff, cutoff, limit=1000)[0]
            results[i, j, 0] = cutoff_integral
            results[i, j, 1] = complete_integral * (1 - epsilon)

    color = px.colors.qualitative.T10[0]
    fig = make_subplots(
        rows=1, cols=len(epsilons), subplot_titles=[f"ε = {epsilon}" for epsilon in epsilons]
    )
    for i, epsilon in enumerate(epsilons):
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=1 - results[i, :, 1] / results[i, :, 0],
                mode="lines",
                line_width=1.5,
                line_color=color,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            range=[-epsilon * 1.05, epsilon * 1.05],
            tickmode="array",
            tickvals=[-epsilon, 0, epsilon],
            ticktext=["-ε", "0", "ε"],
            row=1,
            col=i + 1,
        )
    fig.update_yaxes(title_text="Relative error", row=1, col=1)
    fig.update_xaxes(title_text="Frequency ν")
    fig.update_layout(showlegend=False)
    fig.write_html("data/cutoff.html")
    fig.update_layout(width=650, height=200, margin=dict(t=12, l=0, r=0, b=0))
    font_size = 9
    fig.update_layout(font_size=font_size, legend_font_size=font_size)
    fig.update_annotations(font_size=font_size)
    tickfont_size = int(0.8 * font_size)
    fig.update_xaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_yaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_coloraxes(colorbar_tickfont_size=font_size)
    fig.write_image("data/cutoff.svg")


if __name__ == "__main__":
    main()
