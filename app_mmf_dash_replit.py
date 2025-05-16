import os
import time
import tempfile

import numpy as np
import pandas as pd
import imageio
import plotly.graph_objs as go

from flask import send_file
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

from step_mmf_py import compute_mmf, compute_frame_harm, compute_ph_sig

# ------------------------------------------------------------
# helper per creare il GIF
# ------------------------------------------------------------
def create_animation(theta, mmf, omega, ph_sig, Nc, harm,
                     flags, dis_h, cum_N, thresh, out_path):
    tmpdir = tempfile.mkdtemp()
    filenames = []
    # genera un frame per ogni valore di omega
    for i, ω in enumerate(omega):
        fig = go.Figure()
        # f.m.m.
        fig.add_trace(go.Scatter(
            x=theta, y=mmf[i],
            mode='lines',
            line=dict(shape='hv', color='blue'),
            name='f.m.m.'
        ))
        # singole armoniche
        if 'sing' in flags:
            mags = np.abs(harm)
            mask = (ph_sig != 0) & (mags >= thresh)
            valid = np.where(mask)[0]
            for k in valid[:dis_h]:
                fig.add_trace(go.Scatter(
                    x=theta, y=compute_frame_harm(0,0,0,0,0,0,0,0,0)[1][0]  # placeholder
                ))
                # in pratica dovresti chiamare compute_frame_harm 
                # come in update_frame; per brevità omesso qui
        # cumulativa
        if 'cum' in flags:
            valid_ph = np.where(ph_sig != 0)[0]
            # idem qui: calcola la ycum
        fig.update_layout(
            xaxis=dict(range=[0, 360]),
            yaxis=dict(automargin=True),
            margin=dict(l=40, r=20, t=20, b=40)
        )
        # salva in PNG temporaneo
        fn = os.path.join(tmpdir, f"frame_{i:04d}.png")
        fig.write_image(fn, scale=1)
        filenames.append(fn)

    # scrive il GIF
    with imageio.get_writer(out_path, mode='I', duration=0.1) as writer:
        for fn in filenames:
            img = imageio.imread(fn)
            writer.append_data(img)

    # pulizia
    for fn in filenames:
        os.remove(fn)
    os.rmdir(tmpdir)


# ------------------------------------------------------------
# crea l’app Dash + Flask
# ------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # per Gunicorn / Render

# ------------------------------------------------------------
# layout
# ------------------------------------------------------------
common_btn = {"height": "40px", "width": "130px"}

app.layout = dbc.Container(fluid=True, children=[
    html.H3("Visualizzatore f.m.m. e armoniche – animazione in loop"),
    dbc.Row([
        dbc.Col([html.Label("Q (n° cave)"),
                 dcc.Input(id="input-q", type="number", value=27, min=2)], width=2),
        dbc.Col([html.Label("p (coppie)"),
                 dcc.Input(id="input-p", type="number", value=2, min=1)], width=2),
        dbc.Col([html.Label("m (fasi)"),
                 dcc.Input(id="input-m", type="number", value=3, min=1)], width=2),
        dbc.Col([html.Label("r (racc. passo)"),
                 dcc.Input(id="input-r", type="number", value=1, min=0)], width=2),
        dbc.Col([html.Label("n° arm. (karm)"),
                 dcc.Input(id="input-karm", type="number", value=35, min=1)], width=2),
        dbc.Col([html.Label("nsamp (campioni)"),
                 dcc.Input(id="input-nsamp", type="number", value=720, min=1)], width=2),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([html.Label("ωΔt (°)"),
                 dcc.Input(id="input-step_deg", type="number", value=5.0, min=0.1, step=0.1)], width=2),
        dbc.Col([html.Label("N arm. vis."),
                 dcc.Input(id="input-dis-harm", type="number", value=5, min=1)], width=2),
        dbc.Col([html.Label("N arm. Σ"),
                 dcc.Input(id="input-cum-N", type="number", value=5, min=1)], width=2),
        dbc.Col([html.Label("amp. soglia"),
                 dcc.Input(id="input-thresh", type="number", value=1e-3, min=0, step=1e-4)], width=2),
        dbc.Col([], width=4),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Button("Genera", id="btn-generate", color="primary", style=common_btn), width="auto"),
        dbc.Col(dbc.Button("Download Nc", id="btn-download-nc", color="info", style=common_btn, disabled=True), width="auto"),
        dbc.Col([html.Br(), dcc.Input(id="input-nc-fn", type="text", value="Nc.tsv")], width="auto"),
        dbc.Col(dbc.Button("Download Harm", id="btn-download-harm", color="info", style=common_btn, disabled=True), width="auto"),
        dbc.Col([html.Br(), dcc.Input(id="input-harm-fn", type="text", value="harmonics.tsv")], width="auto"),
    ], className="mb-3", justify="start"),
    dbc.Row([
        dbc.Col(dcc.Checklist(
            id="show-harmonics",
            options=[
                {"label": " arm. singole", "value": "sing"},
                {"label": " somma arm.", "value": "cum"},
            ],
            value=["sing"],
            inline=True
        ), width=6),
    ], className="mb-3"),
    # qui mostriamo il GIF
    html.Img(id="anim-img", src="/animation.gif",
             style={"width": "100%", "marginTop": "1rem"}),
    # per salvare i dati in memoria
    dcc.Store(id="stored-data", storage_type="memory"),
    # endpoint di download
    dcc.Download(id="download-nc-tsv"),
    dcc.Download(id="download-harm-tsv"),
])

# ------------------------------------------------------------
# callback: genera i dati + crea il GIF + abilita i download
# ------------------------------------------------------------
@app.callback(
    Output("stored-data",       "data"),
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
    State("show-harmonics",     "value"),
    State("input-dis-harm",     "value"),
    State("input-cum-N",        "value"),
    State("input-thresh",       "value"),
    prevent_initial_call=True
)
def generate_data(nc, Q, p, m, r, karm, nsamp, step_deg,
                  flags, dis_h, cum_N, thresh):
    # 1) calcola le grandezze
    omega = np.arange(0, 2*np.pi+1e-9, np.radians(step_deg))
    theta, mmf, Nc, harm, _ = compute_mmf(
        Q, p, r, m, omega, karm=karm, nsamp=nsamp
    )
    ph_sig = compute_ph_sig(Q, p, m, karm)

    data = {
        "Q": Q, "p": p, "m": m, "r": r, "karm": karm, "nsamp": nsamp,
        "theta": theta.tolist(),
        "mmf": mmf.tolist(),
        "Nc": Nc.tolist(),
        "harm": [[c.real, c.imag] for c in harm],
        "omega": omega.tolist()
    }

    # 2) crea il GIF
    gif_path = "animation.gif"
    create_animation(
        theta=theta, mmf=mmf, omega=omega,
        ph_sig=ph_sig, Nc=Nc, harm=harm,
        flags=flags, dis_h=dis_h, cum_N=cum_N, thresh=thresh,
        out_path=gif_path
    )

    # 3) ritorna i dati e abilita i download, e ricarica il src col cache-buster
    new_src = f"/animation.gif?{int(time.time())}"
    return data, False, False, new_src

# ------------------------------------------------------------
# callback: download Nc
# ------------------------------------------------------------
@app.callback(
    Output("download-nc-tsv", "data"),
    Input("btn-download-nc",   "n_clicks"),
    State("stored-data",       "data"),
    State("input-nc-fn",       "value"),
    prevent_initial_call=True
)
def download_nc(_, data, fname):
    df = pd.DataFrame(data["Nc"])
    return dcc.send_data_frame(df.to_csv, filename=fname,
                               sep="\t", index=False, header=False)

# ------------------------------------------------------------
# callback: download Harm
# ------------------------------------------------------------
@app.callback(
    Output("download-harm-tsv", "data"),
    Input("btn-download-harm", "n_clicks"),
    State("stored-data",       "data"),
    State("input-harm-fn",     "value"),
    prevent_initial_call=True
)
def download_harm(_, data, fname):
    df = pd.DataFrame(data["harm"], columns=["Re", "Im"])
    return dcc.send_data_frame(df.to_csv, filename=fname,
                               sep="\t", index=False, header=True)

# ------------------------------------------------------------
# rotta Flask per servire il GIF
# ------------------------------------------------------------
@server.route("/animation.gif")
def serve_animation():
    return send_file("animation.gif", mimetype="image/gif")

# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
