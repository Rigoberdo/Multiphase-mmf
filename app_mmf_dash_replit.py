import time
import io
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd

from step_mmf_py import compute_mmf, compute_frame_harm, compute_ph_sig

# Path dove salviamo l'animazione
ANIM_PATH = "animation.gif"

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

common_btn = {"height": "40px", "width": "130px"}

app.layout = dbc.Container([
    html.H3("Visualizzatore f.m.m. e armoniche – animazione in loop"),
    # INPUT
    dbc.Row([
        dbc.Col(html.Label("Q (n° cave)"), width=2),
        dbc.Col(dcc.Input(id="input-q", type="number", value=27, min=2), width=2),
        dbc.Col(html.Label("p (coppie)"), width=2),
        dbc.Col(dcc.Input(id="input-p", type="number", value=2, min=1), width=2),
        dbc.Col(html.Label("m (fasi)"), width=2),
        dbc.Col(dcc.Input(id="input-m", type="number", value=3, min=1), width=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Label("r (racc. passo)"), width=2),
        dbc.Col(dcc.Input(id="input-r", type="number", value=1, min=0), width=2),
        dbc.Col(html.Label("n° arm."), width=2),
        dbc.Col(dcc.Input(id="input-karm", type="number", value=35, min=1), width=2),
        dbc.Col(html.Label("nsamp (campioni)"), width=2),
        dbc.Col(dcc.Input(id="input-nsamp", type="number", value=720, min=1), width=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Label("ωΔt (°)"), width=2),
        dbc.Col(dcc.Input(id="input-step_deg", type="number", value=5.0, step=0.1, min=0.1), width=2),
        dbc.Col(html.Label("N arm. vis."), width=2),
        dbc.Col(dcc.Input(id="input-dis-harm", type="number", value=5, min=1), width=2),
        dbc.Col(html.Label("N arm. Σ"), width=2),
        dbc.Col(dcc.Input(id="input-cum-N", type="number", value=5, min=1), width=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Label("amp. soglia"), width=2),
        dbc.Col(dcc.Input(id="input-thresh", type="number", value=1e-3, step=1e-4, min=0), width=2),
        dbc.Col([], width=8),
    ], className="mb-3"),

    # PULSANTI
    dbc.Row([
        dbc.Col(dbc.Button("Genera",        id="btn-generate",      color="primary", style=common_btn), width="auto"),
        dbc.Col(dbc.Button("Play/Pause",    id="btn-play",          color="success", style=common_btn, disabled=True), width="auto"),
        dbc.Col(dbc.Button("Stop",          id="btn-stop",          color="danger",  style=common_btn, disabled=True), width="auto"),
        dbc.Col(dbc.Button("Download Nc",   id="btn-download-nc",   color="info",    style=common_btn, disabled=True), width="auto"),
        dbc.Col(dbc.Button("Download Harm", id="btn-download-harm", color="info",    style=common_btn, disabled=True), width="auto"),
    ], className="mb-4", justify="start"),

    # IMG PER ANIMAZIONE
    html.Img(id="anim-img", src="/animation.gif", style={"width":"100%", "margin-bottom":"1rem"}),

    # CHECKLIST ARMONICHE
    dcc.Checklist(
        id="show-harmonics",
        options=[
            {"label":" Singole armoniche","value":"sing"},
            {"label":" Somma cumulativa","value":"cum"},
        ],
        value=["sing"], inline=True
    ),

    # GRAFICO (in modalità pause mostra l’ultima frame)
    dcc.Graph(id="plot-fmm"),

    # INTERVAL
    dcc.Interval(id="interval", interval=200, disabled=True),

    # STORE DATI
    dcc.Store(id="stored-data", storage_type="memory"),
    dcc.Download(id="download-nc-tsv"),
    dcc.Download(id="download-harm-tsv"),
], fluid=True)


def create_animation(theta, mmf, amp, filepath, fps=10):
    """
    Genera e salva una GIF in `filepath` mostrando mmf.shape[0] frame,
    uno per ogni riga di `mmf`, con asse y fisso su [-amp, +amp].
    """
    frames = []
    for frame in range(mmf.shape[0]):
        # creiamo un grafico matplotlib
        plt.figure(figsize=(6,3))
        plt.step(theta, mmf[frame], where='post', color='blue')
        plt.ylim(-amp, amp)
        plt.xlim(0, 360)
        plt.xlabel('θ (°)')
        plt.ylabel('f.m.m. (A)')
        plt.tight_layout()

        # salva su buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # carica come immagine
        img = Image.open(buf)
        frames.append(np.array(img))

    # scrivi la GIF
    imageio.mimsave(filepath, frames, fps=fps)


@app.callback(
    Output("stored-data",       "data"),
    Output("interval",          "disabled"),
    Output("btn-play",          "disabled"),
    Output("btn-stop",          "disabled"),
    Output("btn-download-nc",   "disabled"),
    Output("btn-download-harm", "disabled"),
    Output("anim-img",          "src"),
    Input("btn-generate",       "n_clicks"),
    State("input-q",            "value"),
    State("input-p",            "value"),
    State("input-m",            "value"),
    State("input-r",            "value"),
    State("input-karm",         "value"),
    State("input-nsamp",        "value"),
    State("input-step_deg",     "value"),
    prevent_initial_call=True
)
def generate_data(nc, Q, p, m, r, karm, nsamp, step_deg):
    # 1) calcolo omega per un solo periodo
    step_rad = np.radians(step_deg)
    omega_t  = np.arange(0, 2*np.pi+1e-9, step_rad)

    # 2) compute_mmf
    theta, mmf, Nc, harm, _ = compute_mmf(Q, p, r, m, omega_t, karm, nsamp)

    # 3) ampiezza positiva massima per yaxis e GIF
    amp = max(abs(mmf.min()), abs(mmf.max())) * 1.15

    # 4) generazione GIF
    create_animation(theta, mmf, amp, ANIM_PATH, fps=int(1000/200))

    # 5) prepara i dati da memorizzare
    data = {
        "Q":Q, "p":p, "m":m, "r":r, "karm":karm, "nsamp":nsamp,
        "theta": theta.tolist(),
        "mmf":   mmf.tolist(),
        "Nc":    Nc.tolist(),
        "harm":  [[c.real, c.imag] for c in harm],
        "omega": omega_t.tolist()
    }

    # 6) forza il reload dell'immagine con un timestamp
    new_src = f"/animation.gif?{int(time.time())}"

    # abilito tutti i controlli
    return data, True, False, True, False, False, new_src


@app.callback(
    Output("plot-fmm", "figure"),
    Input("interval",        "n_intervals"),
    State("stored-data",     "data"),
    State("show-harmonics",  "value"),
    State("input-dis-harm",  "value"),
    State("input-cum-N",     "value"),
    State("input-thresh",    "value"),
    prevent_initial_call=True
)
def update_frame(n, data, flags, dis_h, cum_N, thresh):
    theta = np.array(data["theta"])
    mmf   = np.array(data["mmf"])
    harm  = np.array([complex(r_,i_) for r_,i_ in data["harm"]])
    omega = np.array(data["omega"])
    idx   = n % len(omega)

    ph_sig      = compute_ph_sig(data["Q"], data["p"], data["m"], data["karm"])
    theta_f, f_frame = compute_frame_harm(
        data["Q"], data["p"], data["r"], data["m"],
        omega[idx], data["karm"], data["nsamp"],
        np.array(data["Nc"]), harm
    )

    fig = go.Figure()
    # f.m.m.
    fig.add_trace(go.Scatter(x=theta, y=mmf[idx],
                             mode='lines',
                             line=dict(shape='hv', color='blue'),
                             name='f.m.m.'))
    # armoniche singole
    if 'sing' in flags:
        mask  = (ph_sig != 0) & (np.abs(harm) >= thresh)
        valid = np.where(mask)[0]
        for k in valid[:dis_h]:
            fig.add_trace(go.Scatter(
                x=theta_f, y=f_frame[k,:],
                mode='lines', name=f'H{k+1}'
            ))
    # somma cumulativa
    if 'cum' in flags:
        valid = np.where(ph_sig != 0)[0]
        ycum  = np.sum(f_frame[valid[:cum_N],:], axis=0)
        fig.add_trace(go.Scatter(
            x=theta_f, y=ycum,
            mode='lines',
            line=dict(width=3, color='crimson'),
            name='Σ1..N'
        ))

    fig.update_layout(
        xaxis=dict(title='θ (°)', tick0=0, dtick=30, range=[0,360]),
        yaxis=dict(title='f.m.m. (A)', range=[-amp, amp]),
        transition_duration=0,
        uirevision='keep'
    )
    return fig


@app.callback(
    Output("download-nc-tsv",   "data"),
    Input ("btn-download-nc",   "n_clicks"),
    State ("stored-data",       "data"),
    prevent_initial_call=True
)
def download_nc(_, data):
    df = pd.DataFrame(data["Nc"])
    return dcc.send_data_frame(df.to_csv,
                               filename="Nc.tsv",
                               sep="\t", index=False, header=False)


@app.callback(
    Output("download-harm-tsv", "data"),
    Input ("btn-download-harm", "n_clicks"),
    State ("stored-data",       "data"),
    prevent_initial_call=True
)
def download_harm(_, data):
    df = pd.DataFrame(data["harm"], columns=["Re","Im"])
    return dcc.send_data_frame(df.to_csv,
                               filename="harmonics.tsv",
                               sep="\t", index=False, header=True)


if __name__ == "__main__":
    app.run(debug=True)
