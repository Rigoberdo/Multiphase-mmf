import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from step_mmf_py import compute_mmf, compute_frame_harm, compute_ph_sig

common_btn = {"height": "40px", "width": "130px"}

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Container([
    html.H3("F.m.m. e armoniche in avvolgimento multifase"),
    # PRIMA RIGA DI INPUT
    dbc.Row([
        dbc.Col([html.Label("Q (n° cave)"),
                 dcc.Input(id="input-q", type="number", value=27, min=2)], width=2),
        dbc.Col([html.Label("p (coppie)"),
                 dcc.Input(id="input-p", type="number", value=2, min=1)], width=2),
        dbc.Col([html.Label("m (fasi)"),
                 dcc.Input(id="input-m", type="number", value=3, min=1)], width=2),
        dbc.Col([html.Label("r (racc. passo)"),
                 dcc.Input(id="input-r", type="number", value=1, min=0)], width=2),
        dbc.Col([html.Label("Amp totale (A)"),
                 dcc.Input(id="input-ampfili", type="number", value=1, min=0)], width=2),
    ], className="mb-3"),
    # SECONDA RIGA DI INPUT
    dbc.Row([
        dbc.Col([html.Label("karm (n° arm.)"),
                 dcc.Input(id="input-karm", type="number", value=35, min=1)], width=2),
        dbc.Col([html.Label("nsamp (campioni)"),
                 dcc.Input(id="input-nsamp", type="number", value=720, min=1)], width=2),
        dbc.Col([html.Label("Δθ (°)"),
                 dcc.Input(id="input-step_deg", type="number", value=5.0, step=0.1, min=0.1)], width=2),
        dbc.Col([html.Label("N harm vis."),
                 dcc.Input(id="input-dis-harm", type="number", value=5, min=1)], width=2),
        dbc.Col([html.Label("N harm Σ"),
                 dcc.Input(id="input-cum-N", type="number", value=5, min=1)], width=2),
        dbc.Col([html.Label("soglia amp."),
                 dcc.Input(id="input-thresh", type="number", value=1e-3, step=1e-4, min=0)], width=2),
    ], className="mb-4"),
    # BOTTONE GENERA + PLAY/PAUSE + STOP + DOWNLOAD
    dbc.Row([
        dbc.Col(dbc.Button("Genera",        id="btn-generate",    color="primary", style=common_btn), width="auto"),
        dbc.Col(dbc.Button("Play/Pause",    id="btn-play",        color="success", style=common_btn, disabled=True), width="auto"),
        dbc.Col(dbc.Button("Stop",          id="btn-stop",        color="danger",  style=common_btn, disabled=True), width="auto"),
        dbc.Col(dbc.Button("matr. avv.",    id="btn-download-nc", color="info",    style=common_btn, disabled=True),
                width="auto"),
        dbc.Col(dcc.Input(id="input-nc-fn",  type="text", value="Nc.tsv"), width="auto"),
        dbc.Col(dbc.Button("spettro arm.",  id="btn-download-harm", color="info", style=common_btn, disabled=True),
                width="auto"),
        dbc.Col(dcc.Input(id="input-harm-fn", type="text", value="harmonics.tsv"), width="auto"),
    ], className="mb-3", justify="start"),
    # CHECKLIST ARMONICHE
    dbc.Row([
        dbc.Col(dcc.Checklist(
            id="show-harmonics",
            options=[
                {"label": " arm. sing.", "value": "sing"},
                {"label": " somma cum.", "value": "cum"},
            ],
            value=["sing"], inline=True
        ), width=4),
    ], className="mb-3"),
    # GRAFICO + INTERVALLO
    dcc.Graph(id="plot-fmm"),
    dcc.Interval(id="interval", interval=400, disabled=True),
    # STORE DATI + DOWNLOAD COMPONENTS
    dcc.Store(id="stored-data", storage_type="memory"),
    dcc.Download(id="download-nc-tsv"),
    dcc.Download(id="download-harm-tsv"),
], fluid=True)


# --- 1) GENERA DATI e abilita DOWNLOAD ---
@app.callback(
    Output("stored-data",       "data"),
    Output("btn-download-nc",   "disabled"),
    Output("btn-download-harm", "disabled"),
    Input("btn-generate",       "n_clicks"),
    State("input-q",            "value"),
    State("input-p",            "value"),
    State("input-m",            "value"),
    State("input-r",            "value"),
    State("input-ampfili",      "value"),
    State("input-karm",         "value"),
    State("input-nsamp",        "value"),
    State("input-step_deg",     "value"),
    prevent_initial_call=True
)
def generate_data(nc, Q, p, m, r, ampfili, karm, nsamp, step_deg):
    omega = np.arange(0, 2*np.pi+1e-9, np.radians(step_deg))
    theta, mmf, Nc, harm, omega = compute_mmf(
        ampfili, Q, p, r, m, omega, karm=karm, nsamp=nsamp
    )
    data = {
        "Q":Q, "p":p, "m":m, "r":r,
        "ampfili":ampfili,
        "karm":karm, "nsamp":nsamp,
        "theta":theta.tolist(),
        "mmf":mmf.tolist(),
        "Nc":Nc.tolist(),
        "harm":[[c.real, c.imag] for c in harm],
        "omega":omega.tolist()
    }
    # abilito i pulsanti di download
    return data, False, False


# --- 2) PLAY/PAUSE/STOP (UNICO CALLBACK) ---
@app.callback(
    Output("interval", "disabled"),
    Output("btn-play", "disabled"),
    Output("btn-stop", "disabled"),
    Input("btn-generate", "n_clicks"),
    Input("btn-play",     "n_clicks"),
    Input("btn-stop",     "n_clicks"),
    State("interval",     "disabled"),
    prevent_initial_call=True
)
def control_interval(n_gen, n_play, n_stop, disabled):
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "btn-generate":
        # appena genero: stop attivo, play abilitato
        return True, False, True
    elif trig == "btn-play":
        # toggle Play/Pause
        return (not disabled), False, False
    else:
        # Stop
        return True, False, True


# --- 3) AGGIORNA FRAME e RIDISEGNA GRAFICO ---
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
    # estraggo dati
    theta = np.array(data["theta"])
    mmf   = np.array(data["mmf"])
    Nc    = np.array(data["Nc"])
    harm  = np.array([complex(r,i) for r,i in data["harm"]])
    omega = np.array(data["omega"])
    Q, p, m, karm = data["Q"], data["p"], data["m"], data["karm"]
    idx = n % len(omega)

    # fmm a gradini
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta, y=mmf[idx],
        mode='lines', line=dict(shape='hv', color='blue'),
        name='f.m.m.'
    ))

    # armoniche singole
    if 'sing' in flags:
        ph_sig = compute_ph_sig(Q, p, m, karm)
        mask   = (ph_sig != 0) & (np.abs(harm) >= thresh)
        valid  = np.where(mask)[0]
        theta_f, f_frame = compute_frame_harm(
            Q, p, data["r"], m, omega[idx], karm, data["nsamp"], Nc, harm
        )
        for k in valid[:dis_h]:
            fig.add_trace(go.Scatter(
                x=theta_f, y=f_frame[k, :],
                mode='lines', name=f'H{k+1}'
            ))

    # somma cumulativa
    if 'cum' in flags:
        ph_sig = compute_ph_sig(Q, p, m, karm)
        valid  = np.where(ph_sig != 0)[0]
        theta_f, f_frame = compute_frame_harm(
            Q, p, data["r"], m, omega[idx], karm, data["nsamp"], Nc, harm
        )
        ycum = np.sum(f_frame[valid[:cum_N], :], axis=0)
        fig.add_trace(go.Scatter(
            x=theta_f, y=ycum,
            mode='lines', line=dict(width=3, color='crimson'),
            name='Σ 1..N'
        ))

    fig.update_layout(
        xaxis=dict(title='θ (°)', tick0=0, dtick=30, range=[0,360]),
        yaxis=dict(title='f.m.m. (A)'),
        transition_duration=0, uirevision='constant'
    )
    return fig


# --- 4) DOWNLOAD FILES Nc e Harm ---
@app.callback(
    Output("download-nc-tsv", "data"),
    Input("btn-download-nc", "n_clicks"),
    State("stored-data", "data"),
    State("input-nc-fn",    "value"),
    prevent_initial_call=True
)
def download_nc(_, data, fname):
    df = pd.DataFrame(data["Nc"])
    return dcc.send_data_frame(df.to_csv,
                               filename=fname,
                               sep="\t", index=False, header=False)

@app.callback(
    Output("download-harm-tsv", "data"),
    Input("btn-download-harm", "n_clicks"),
    State("stored-data",       "data"),
    State("input-harm-fn",     "value"),
    prevent_initial_call=True
)
def download_harm(_, data, fname):
    df = pd.DataFrame(data["harm"], columns=["Re","Im"])
    return dcc.send_data_frame(df.to_csv,
                               filename=fname,
                               sep="\t", index=False, header=True)


#if __name__ == "__main__":
#    app.run(debug=True)
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
