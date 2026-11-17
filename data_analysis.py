import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path(__file__).parent / "lab_data.csv"

# Columns: Run label, Orifice Plate (cm H2O), T1..T20 (cm Hg)
raw = np.genfromtxt(DATA_FILE, delimiter=",", skip_header=1, dtype=str)
runs = raw[:, 0]                         # e.g. "Run 1"
data = raw[:, 1:].astype(float)
orifice = data[:, 0]                     # orifice plate pressure drop, cm H2O, shape (4,)
tappings = data[:, 1:]                   # manometer LEVELS T1..T20, cm Hg, shape (4, 20)
                                         # T1 = plenum chamber, T19 = nozzle exit plane

###CONSTANTS###
rho_Hg = 13533.6 #kg/m3
rho_H2O = 1008 #kg/m3
p_atm_reading = 30.09 #inHg (barometer)
p_atm = p_atm_reading * 2.54e-2 * 9.81 * rho_Hg #Pa
T_atm = 20.1 #degC
T_atm_K = T_atm + 273.15 #K
gamma = 1.4
#air density = PM/RT
rho_air = p_atm * 28.97e-3 / (8.314 * T_atm_K) #kg/m3, M in kg/mol
D_orifice = 17.47e-3 #m
C_d = 0.62
p_po = 0.5283 #p/p0 for M = 1 (gamma = 1.4)
exit_idx = 18 #T19 = nozzle exit plane (0-based index)

#Convert the pressure reading from cm of Hg or H2O into Pa
def convert(data, rho):
    #P = p_reading (cmHg or cmH2O) * 1e-2 (to get into metres) * g * density
    pressure = data * 1e-2 * 9.81 * rho
    return pressure

def mach_from_ratio(p_over_p0):
    #Isentropic: p0/p = (1 + (gamma-1)/2 M^2)^(gamma/(gamma-1)), rearranged for M
    ratio = np.minimum(p_over_p0, 1.0)   # guard against p slightly above p0 (gives M = 0)
    return np.sqrt(2 / (gamma - 1) * ((1 / ratio)**((gamma - 1) / gamma) - 1))

###ORIFICE CALIBRATION###
orifice_Pa = convert(orifice, rho_H2O) #pressure drop between atmosphere and orifice plane
#Use Bernoulli to find the orifice velocity (incompressible: V ~ 30 m/s, M ~ 0.09)
V_orifice = np.sqrt(2 * orifice_Pa / rho_air) #m/s
m_dot_uncorrected = rho_air * V_orifice * np.pi * (D_orifice**2) / 4 #kg/s
m_dot = m_dot_uncorrected * C_d #kg/s

###CONVERT MANOMETER LEVELS TO ABSOLUTE PRESSURE###
#The mercury bank gives LEVELS, not absolute pressures: p = p_atm - rho_Hg g (h_atm - h),
#where h_atm is the level of the tubes open to atmosphere.
#ASSUMPTION: h_atm was not recorded, so it is estimated for each run from the plenum (T1).
#The plenum pressure is taken as atmospheric minus the orifice drop (the orifice jet does
#not recover its dynamic head in the plenum), so h_atm = h_T1 + orifice drop in cmHg.
#Estimated per run because the reservoir level of the bank shifts slightly between runs.
#Check: this gives a throat area from the mass flow of ~24 mm^2, as quoted in the handout.
h_atm = tappings[:, 0] + orifice * rho_H2O / rho_Hg            #cm Hg, shape (4,)
tappings_Pa = p_atm - convert(h_atm[:, None] - tappings, rho_Hg) #absolute pressure, Pa
p1 = tappings_Pa[:, 0]                                          #plenum pressure, Pa

#Pressure ratio p/p1 at each tapping (p1 = plenum = stagnation pressure)
abs_p_ratio = tappings_Pa / p1[:, None]

###MACH NUMBERS (isentropic, p0 = p1, T0 = T_room)###
#Only valid where the flow is isentropic, i.e. upstream of any shock.
M = mach_from_ratio(abs_p_ratio) #shape (4, 20)

np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("Mass flow (kg/s):", m_dot)
print("Estimated h_atm (cmHg):", h_atm)
print("Plenum pressure p1 (kPa):", p1 / 1e3)
print("Mach number at each tapping (rows = runs, columns = T1..T20):")
print(M)

###THROAT POSITION###
#For the choked runs (2-4) the flow is sonic at the throat, so the throat is where
#p/p1 first falls through 0.5283. Interpolate between the two tappings either side.
def first_crossing(ratio):
    i = np.argmax(ratio < p_po)                      #first tapping below p_po (0-based)
    return i + (ratio[i - 1] - p_po) / (ratio[i - 1] - ratio[i])   #tapping number

throat_each = np.array([first_crossing(abs_p_ratio[k]) for k in (1, 2, 3)])
throat_pos = throat_each.mean()
throat_lo = int(np.floor(throat_pos))                #tapping just upstream of the throat
nearest_idx = int(round(throat_pos)) - 1             #tapping nearest the throat (0-based)
print(f"Throat (p/p1 = {p_po}) at tapping {throat_pos:.2f} (runs 2-4: {throat_each})")

#Sanity check from handout: M at the tapping nearest the throat within 5% of 1
for k in (1, 2, 3):
    m = M[k, nearest_idx]
    print(f"{runs[k]}: M at T{nearest_idx + 1} = {m:.3f} ({100 * (m - 1):+.1f}% from M = 1)")

#Throat and exit Mach numbers for runs 1, 2 and 4
tap = np.arange(1, tappings.shape[1] + 1)
for k in (0, 1, 3):
    M_throat = np.interp(throat_pos, tap, M[k])
    print(f"{runs[k]}: M_throat = {M_throat:.2f}, M_exit (T19) = {M[k, exit_idx]:.2f}")

###RUN 3 SHOCK###
abs_p_run3 = abs_p_ratio[2]
#Shock position: where p/p1 rises back through p_po on the downstream side
#(search after the pressure minimum so we skip the first crossing near the throat)
i_min = np.argmin(abs_p_run3)
for i in range(i_min + 1, len(abs_p_run3)):
    if abs_p_run3[i] >= p_po:
        shock_idx = i - 1               # last tapping before p/p1 reaches p_po (0-based)
        break
shock_tapping = shock_idx + 1

#Run 3's wall pressure at the shock is already raised by the smeared shock due to the boundary layer, so it
#can't give M1. Upstream of the shock the core flow is the same as Run 4 (same nozzle,
#same p0, choked throat), and Run 4 is still supersonic here -> use Run 4 values.
M1 = M[3, shock_idx]                    # Mach just before the shock
p1_shock = tappings_Pa[3, shock_idx]    # static pressure just before the shock

M_s = np.sqrt((1 + (gamma - 1) / 2 * M1**2) / (gamma * M1**2 - (gamma - 1) / 2))  # Mach after shock
ps_p1 = 1 + ((2 * gamma) / (gamma + 1)) * (M1**2 - 1)  # static pressure ratio across shock
ps_shock = p1_shock * ps_p1                            # predicted pressure after the shock

p_run3_measured = tappings_Pa[2, shock_idx]  # Run 3 wall pressure there
measured_jump = tappings_Pa[2, i_min:].max() - tappings_Pa[2, i_min]

print(f"Run 3 shock position: tapping {shock_tapping}")
print(f"M1 = {M1:.2f}, Ms = {M_s:.2f}, ps/p1 = {ps_p1:.2f}")
print(f"p1 = {p1_shock/1e3:.1f} kPa -> predicted ps = {ps_shock/1e3:.1f} kPa, "
      f"Run 3 measured = {p_run3_measured/1e3:.1f} kPa")
print(f"Predicted jump = {(ps_shock - p1_shock)/1e3:.1f} kPa, "
      f"measured (min T{i_min + 1} -> max downstream) = {measured_jump/1e3:.1f} kPa")

###PREVIOUS (INCORRECT) METHOD - kept for reference / comparison###
#Treated the manometer LEVELS as absolute pressures (p = rho_Hg g h) and used p0 = p_atm
#for the Mach numbers. This puts the plenum at ~81 kPa instead of ~101 kPa, which gives a
#throat area from the mass flow of ~29.5 mm^2 instead of the ~24 mm^2 quoted in the handout.
#Variables use an _old suffix so they don't overwrite the corrected results above.
print("\n--- PREVIOUS (INCORRECT) METHOD ---")
tappings_Pa_old = convert(tappings, rho_Hg)                      #levels treated as absolute
abs_p_ratio_old = tappings_Pa_old / tappings_Pa_old[:, [0]]      #p/p_T1
M_old = np.sqrt(2 / (gamma - 1) * ((p_atm / tappings_Pa_old)**((gamma - 1) / gamma) - 1))  #p0 = p_atm
print("Mach number at each tapping (old method):")
print(M_old)
#Output (values downstream of a shock are NOT valid):
#        T1   T2   T3   T4   T5   T6   T7   T8   T9   T10  T11  T12  T13  T14  T15  T16  T17  T18  T19  T20
#Run 1:  0.59 0.60 0.61 0.61 0.63 0.65 0.70 0.79 0.90 0.94 0.92 0.90 0.88 0.86 0.85 0.84 0.83 0.82 0.81 0.81 (Subsonic)
#Run 2:  0.58 0.60 0.60 0.61 0.63 0.67 0.74 0.88 1.15 1.49 1.47 1.32 1.14 1.10 1.07 1.04 1.02 1.00 0.99 0.97
#Run 3:  0.57 0.59 0.60 0.61 0.63 0.66 0.74 0.88 1.14 1.49 1.70 1.85 1.95 1.89 1.79 1.47 1.26 1.20 1.16 1.12 (Shock between T13 & T14)
#Run 4:  0.57 0.59 0.59 0.60 0.62 0.66 0.73 0.87 1.13 1.48 1.69 1.83 1.94 1.97 2.07 2.21 2.34 2.48 2.45 2.25

#Run 3 shock (old method)
abs_p_run3_old = abs_p_ratio_old[2]
i_min_old = np.argmin(abs_p_run3_old)
for i in range(i_min_old + 1, len(abs_p_run3_old)):
    if abs_p_run3_old[i] >= p_po:
        shock_idx_old = i - 1
        break
M1_old = M_old[3, shock_idx_old]
p1_shock_old = tappings_Pa_old[3, shock_idx_old]
M_s_old = np.sqrt((1 + (gamma - 1) / 2 * M1_old**2) / (gamma * M1_old**2 - (gamma - 1) / 2))
ps_p1_old = 1 + ((2 * gamma) / (gamma + 1)) * (M1_old**2 - 1)
ps_shock_old = p1_shock_old * ps_p1_old
p_run3_measured_old = tappings_Pa_old[2, shock_idx_old]

print(f"Run 3 shock position (old): tapping {shock_idx_old + 1}")
print(f"M1 = {M1_old:.2f}, Ms = {M_s_old:.2f}, ps/p1 = {ps_p1_old:.2f}")
print(f"p1 = {p1_shock_old/1e3:.1f} kPa -> predicted ps = {ps_shock_old/1e3:.1f} kPa, "
      f"Run 3 measured = {p_run3_measured_old/1e3:.1f} kPa")
#Output from shock prediction calculations (old method)
#Shock position: tapping 18
#M1 = 2.48, M2 = 0.51, p2/p1 = 7.02
#p1 = 6.1 kPa -> predicted p2 = 42.9 kPa, Run 3 measured = 41.8 kPa
#Predicted jump = 36.8 kPa, measured (min -> T20) = 32.1 kPa

###SAVE ALL PRINTED RESULTS TO CSV###
#One file, split into labelled sections separated by blank rows (opens cleanly in Excel)
import csv
tap_header = [f"T{t}" for t in range(1, tappings.shape[1] + 1)]
r = lambda x, n=4: round(float(x), n)

with open(Path(__file__).parent / "results.csv", "w", newline="") as f:
    w = csv.writer(f)

    w.writerow(["CORRECTED METHOD - per-run summary"])
    w.writerow(["Run", "Orifice drop (Pa)", "Mass flow (kg/s)", "h_atm (cmHg)", "p1 (kPa)",
                "M_throat (interp.)", "M_exit (T19)", f"M at T{nearest_idx + 1} (sanity)",
                "Deviation from M=1 (%)"])
    for k in range(len(runs)):
        M_thr = np.interp(throat_pos, tap, M[k])
        valid_exit = k != 2                    # Run 3 exit is behind the shock
        choked = k != 0                        # sanity check only applies to choked runs
        w.writerow([runs[k], r(orifice_Pa[k], 2), r(m_dot[k], 6), r(h_atm[k], 3), r(p1[k] / 1e3, 3),
                    r(M_thr, 3) if k != 2 else "", r(M[k, exit_idx], 3) if valid_exit else "",
                    r(M[k, nearest_idx], 3) if choked else "",
                    r(100 * (M[k, nearest_idx] - 1), 1) if choked else ""])
    w.writerow([])

    w.writerow(["Throat position (p/p1 = 0.5283 crossing)"])
    w.writerow(["Run 2", "Run 3", "Run 4", "Mean"])
    w.writerow([r(x, 3) for x in throat_each] + [r(throat_pos, 3)])
    w.writerow([])

    w.writerow(["Mach number at each tapping (corrected; not valid downstream of a shock)"])
    w.writerow(["Run"] + tap_header)
    for k in range(len(runs)):
        w.writerow([runs[k]] + [r(x, 3) for x in M[k]])
    w.writerow([])

    w.writerow(["Run 3 shock (corrected)"])
    w.writerow(["Quantity", "Value", "Unit"])
    for row in [("Shock position", shock_tapping, "tapping"),
                ("M1 (from Run 4)", r(M1, 3), "-"),
                ("Ms (after shock)", r(M_s, 3), "-"),
                ("ps/p1", r(ps_p1, 3), "-"),
                ("p1 before shock", r(p1_shock / 1e3, 2), "kPa"),
                ("Predicted ps", r(ps_shock / 1e3, 2), "kPa"),
                (f"Run 3 measured at T{shock_tapping}", r(p_run3_measured / 1e3, 2), "kPa"),
                ("Predicted jump", r((ps_shock - p1_shock) / 1e3, 2), "kPa"),
                (f"Measured jump (min T{i_min + 1} -> max downstream)", r(measured_jump / 1e3, 2), "kPa")]:
        w.writerow(row)
    w.writerow([])

    w.writerow(["PREVIOUS (INCORRECT) METHOD - Mach number at each tapping (p0 = p_atm)"])
    w.writerow(["Run"] + tap_header)
    for k in range(len(runs)):
        w.writerow([runs[k]] + [r(x, 3) for x in M_old[k]])
    w.writerow([])

    w.writerow(["Run 3 shock (previous method)"])
    w.writerow(["Quantity", "Value", "Unit"])
    for row in [("Shock position", shock_idx_old + 1, "tapping"),
                ("M1", r(M1_old, 3), "-"),
                ("Ms (after shock)", r(M_s_old, 3), "-"),
                ("ps/p1", r(ps_p1_old, 3), "-"),
                ("p1 before shock", r(p1_shock_old / 1e3, 2), "kPa"),
                ("Predicted ps", r(ps_shock_old / 1e3, 2), "kPa"),
                (f"Run 3 measured at T{shock_idx_old + 1}", r(p_run3_measured_old / 1e3, 2), "kPa")]:
        w.writerow(row)
print("\nResults saved to results.csv")

###PLOT: pressure ratio along the nozzle###
tapping_no = tap  # 1..20
colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
markers = ["D", "s", "^", "."]

fig, ax = plt.subplots(figsize=(8, 5))
# Shaded throat region (between the two tappings either side of the sonic point)
ax.axvspan(throat_lo, throat_lo + 1, color="0.85", alpha=0.6, zorder=0, linewidth=0)
# Labels sit just above the top of the axes (y in axes fraction), clear of the band/line
ax.text(throat_lo + 0.5, 1.01, "Throat position", ha="center", va="bottom", fontsize=9,
        color="0.3", transform=ax.get_xaxis_transform())
# Expected pressure ratio at the throat (M = 1)
ax.plot(throat_pos, p_po, marker="x", color="0.1", markersize=9, markeredgewidth=1.5, zorder=5)
ax.annotate(f"$p^*/p_1$\n= {p_po}", xy=(throat_pos, p_po), xytext=(throat_pos + 0.5, p_po + 0.035),
            textcoords="data", ha="left", va="bottom", fontsize=8, color="0.1",
            arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.8, shrinkA=0, shrinkB=5),
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
# Nozzle exit at T19
ax.axvline(exit_idx + 1, color="0.5", linestyle="-", linewidth=1, zorder=0)
ax.text(exit_idx + 1, 1.01, "Nozzle exit", ha="center", va="bottom", fontsize=9, color="0.3",
        transform=ax.get_xaxis_transform())
for k in range(len(runs)):
    ax.plot(tapping_no, abs_p_ratio[k], color=colours[k], marker=markers[k],
            linestyle="--", linewidth=1, markersize=6, label=runs[k])

t_up, t_down = throat_lo, throat_lo + 1   # tappings either side of the throat
mach_labels = [
    # (run indices sharing the label, tapping, text offset in points, alignment)
    # Runs 2 and 4 are identical up to T10 (choked), so they share one label there
    ((0,),   t_up,   (10, 14),  "left"),
    ((0,),   t_down, (8, -16),  "left"),
    ((1, 3), t_up,   (-10, 0),  "right"),
    ((1, 3), t_down, (-10, -12), "right"),
    ((0,),   19,     (0, 9),    "center"),
    ((1,),   19,     (0, 9),    "center"),
    ((3,),   19,     (0, -15),  "center"),
]
for ks, t, offset, ha in mach_labels:
    k = ks[0]
    who = f"Run {k + 1}" if len(ks) == 1 else "Runs " + " & ".join(str(j + 1) for j in ks)
    ax.annotate(f"{who}: $M_{{T{t}}}$ = {M[k, t - 1]:.2f}", xy=(t, abs_p_ratio[k, t - 1]),
                xytext=offset, textcoords="offset points", ha=ha, fontsize=8, color="0.2",
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))

ax.set_xlabel("Tapping number")
ax.set_ylabel(r"$p / p_1$")
ax.set_xticks(tapping_no)
ax.set_xlim(0.5, tapping_no[-1] + 0.5)
ax.set_ylim(bottom=0)
ax.grid(True, color="0.9", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
# Extra legend entry (no line or marker) explaining the M_Tn notation
ax.plot([], [], " ", label=r"$M_{Tn}$ = Mach number at pressure tapping $n$")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "pressure_ratio.png", dpi=200)

###PLOT: previous (incorrect) method, for comparison###
#Same layout as above, but using levels-as-absolute pressures (p/p_T1) and p0 = p_atm Mach numbers
fig_old, ax_old = plt.subplots(figsize=(8, 5))
ax_old.axvspan(8, 9, color="0.85", alpha=0.6, zorder=0, linewidth=0)
ax_old.text(8.5, 1.01, "Throat position", ha="center", va="bottom", fontsize=9, color="0.3",
            transform=ax_old.get_xaxis_transform())
ax_old.axvline(exit_idx + 1, color="0.5", linestyle="-", linewidth=1, zorder=0)
ax_old.text(exit_idx + 1, 1.01, "Nozzle exit", ha="center", va="bottom", fontsize=9, color="0.3",
            transform=ax_old.get_xaxis_transform())
for k in range(len(runs)):
    ax_old.plot(tapping_no, abs_p_ratio_old[k], color=colours[k], marker=markers[k],
                linestyle="--", linewidth=1, markersize=6, label=runs[k])

mach_labels_old = [
    # (run indices sharing the label, tapping, text offset in points, alignment)
    ((0,),   8,  (10, 4),   "left"),
    ((1,),   8,  (-10, 0),  "right"),
    ((3,),   8,  (-10, -14), "right"),
    ((0,),   9,  (8, 8),    "left"),
    ((1,),   9,  (10, 2),   "left"),
    ((3,),   9,  (-10, -14), "right"),
    ((0,),   19, (0, 9),    "center"),
    ((1,),   19, (0, 9),    "center"),
    ((3,),   19, (0, -15),  "center"),
]
for ks, t, offset, ha in mach_labels_old:
    k = ks[0]
    ax_old.annotate(f"Run {k + 1}: $M_{{T{t}}}$ = {M_old[k, t - 1]:.2f}",
                    xy=(t, abs_p_ratio_old[k, t - 1]), xytext=offset, textcoords="offset points",
                    ha=ha, fontsize=8, color="0.2",
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))

ax_old.set_title("Previous (incorrect) method: manometer levels treated as absolute pressure",
                 fontsize=10, pad=22)
ax_old.set_xlabel("Tapping number")
ax_old.set_ylabel(r"$p / p_1$")
ax_old.set_xticks(tapping_no)
ax_old.set_xlim(0.5, tapping_no[-1] + 0.5)
ax_old.set_ylim(bottom=0)
ax_old.grid(True, color="0.9", linewidth=0.8)
ax_old.set_axisbelow(True)
for side in ("top", "right"):
    ax_old.spines[side].set_visible(False)
ax_old.plot([], [], " ", label=r"$M_{Tn}$ = Mach number at pressure tapping $n$")
ax_old.legend(frameon=False)
fig_old.tight_layout()
fig_old.savefig(Path(__file__).parent / "pressure_ratio_old.png", dpi=200)

###PLOT: comparison of corrected vs previous method, one panel per run###
fig_cmp, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for k, ax_c in enumerate(axes.flat):
    ax_c.plot(tapping_no, abs_p_ratio[k], color=colours[k], marker=markers[k],
              linestyle="-", linewidth=1.5, markersize=5, label=r"Corrected ($p/p_1$)")
    ax_c.plot(tapping_no, abs_p_ratio_old[k], color=colours[k], marker=markers[k],
              linestyle="--", linewidth=1, markersize=5, alpha=0.45,
              markerfacecolor="none", label=r"Previous ($p/p_1$)")
    ax_c.axvline(exit_idx + 1, color="0.5", linewidth=1, zorder=0)
    ax_c.axvspan(throat_lo, throat_lo + 1, color="0.85", alpha=0.6, zorder=0, linewidth=0)
    # Exit Mach number, corrected vs previous (not for Run 3: exit is behind the shock,
    # so the isentropic Mach number isn't valid there)
    if k != 2:
        ax_c.text(0.98, 0.97, f"$M_{{T19}}$: {M[k, exit_idx]:.2f} (corrected) vs "
                  f"{M_old[k, exit_idx]:.2f} (previous)", transform=ax_c.transAxes,
                  ha="right", va="top", fontsize=8, color="0.2",
                  bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
    ax_c.set_title(runs[k], fontsize=10, loc="left")
    ax_c.set_xticks(tapping_no[::2])
    ax_c.set_xlim(0.5, tapping_no[-1] + 0.5)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(True, color="0.9", linewidth=0.8)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
for ax_c in axes[1]:
    ax_c.set_xlabel("Tapping number")
for ax_c in axes[:, 0]:
    ax_c.set_ylabel("Pressure ratio")
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig_cmp.suptitle("Corrected vs previous method (shaded = corrected throat, line = nozzle exit)",
                 fontsize=11)
fig_cmp.tight_layout()
fig_cmp.savefig(Path(__file__).parent / "pressure_ratio_comparison.png", dpi=200)
