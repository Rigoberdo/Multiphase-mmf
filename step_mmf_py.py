
import numpy as np

def modshift(x, p):
    return x - np.floor((x + p/2) / p) * p

def passo_fase(Q, p, m):
    k = 0
    while ((1 + k * m) * Q) % (p * m) != 0:
        k += 1
    return ((1 + k * m) * Q) // (p * m)

def compute_ph_sig(Q, p, m, karm):
    pf = passo_fase(Q, p, m)
    ph_sig = []
    for k in range(1, karm + 1):
        val = modshift((360 / Q) * k * pf, 360) / (360 / m)
        if np.isclose(val, 1, atol=1e-6):
            ph_sig.append(1)
        elif np.isclose(val, -1, atol=1e-6):
            ph_sig.append(-1)
        else:
            ph_sig.append(0)
    return np.array(ph_sig)

def generate_Nc(Q, p, m, r):
    n = Q // m
    g = 2 * p
    mult = 1 + (m % 2)
    idx = np.arange(1, m + 1)
    pos_keys = np.mod((idx - 1) * mult, 2 * m)
    neg_keys = np.mod(m + (idx - 1) * mult, 2 * m)
    seq_all = np.concatenate([
        np.stack([pos_keys, idx], axis=1),
        np.stack([neg_keys, -idx], axis=1)
    ], axis=0)
    order = np.argsort(seq_all[:,0])
    fasiseq = seq_all[order,1]
    Nc = np.zeros((Q, m))
    k = 0
    g1 = g % (2*m*n)
    gg = g1 - 1
    i = 0
    while i < Q:
        k = (k + gg//n) % (2*m)
        gg = g1 + (gg % n)
        fase = abs(fasiseq[k]) - 1
        segno = np.sign(fasiseq[k])
        Nc[i, int(fase)] = segno
        i += 1
    pitch = Q // (2*p)
    Nc = (Nc - np.roll(Nc, shift=(pitch - r), axis=0)) / 2
    return Nc

def compute_harmonics(Q, p, m, Nc, karm, Imax):
    alpha = np.arange(Q) * 2*np.pi/Q
    ph_sig = compute_ph_sig(Q, p, m, karm)
    harm = np.zeros(karm, dtype=complex)
    for k in range(1, karm+1):
        harm[k-1] = Imax*abs(ph_sig[k-1])*np.dot(Nc[:,0], np.exp(-1j*k*alpha)) / (2*k)
    harm *= m/np.pi
    return harm, ph_sig

def compute_mmf(ampfili,Q, p, r, m, omega_t, karm, nsamp):
    Nc = generate_Nc(Q, p, m, r)
    Imax = ampfili/Q    
    harm, ph_sig = compute_harmonics(Q, p, m, Nc, karm, Imax)
    omega_t = np.atleast_1d(omega_t)
    M = Q + 1

    mmf = np.zeros((len(omega_t), 2*M))
    for ti, ot in enumerate(omega_t):
        I = Imax*np.cos(ot - 2*np.pi*(np.arange(m))/m)
        mmf_t = np.zeros(2*M)
        sum_mmf = 0.0
        for nc in range(1, M+1):
            Iq = np.sum(Nc[nc-1,:]*I) if nc<=Q else np.sum(Nc[0,:]*I)
            if nc > 1:
                mmf_t[2*nc-2] = mmf_t[2*nc-3]
            mmf_t[2*nc-1] = mmf_t[2*nc-2] + Iq
            if nc <= Q:
                sum_mmf += mmf_t[2*nc-1]
        mmf_t -= sum_mmf/Q
        mmf[ti,:] = mmf_t

    theta_deg = np.zeros(2*M)
    delta = 360.0/Q
    for nc in range(1, M+1):
        theta_deg[2*nc-2] = delta*(nc-1)
        theta_deg[2*nc-1] = delta*(nc-1)

    return theta_deg, mmf, Nc, harm, omega_t

def compute_frame_harm(Q, p, r, m, omega, karm, nsamp, Nc, harm):
    ph_sig = compute_ph_sig(Q, p, m, karm)
    theta_f_deg = np.linspace(0, 360, nsamp+1)
    theta_f_rad = np.radians(theta_f_deg)
    f_frame = np.zeros((karm, nsamp+1))
    for k in range(1, karm+1):
        if ph_sig[k-1] != 0:
            amp   = abs(harm[k-1])
            phase = np.angle(harm[k-1])
            # la k moltiplica solo theta_f_rad, non omega
            f_frame[k-1,:] = amp * np.sin(
                k*theta_f_rad 
                - ph_sig[k-1]*omega 
                + phase
            )
    return theta_f_deg, f_frame
