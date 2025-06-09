import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from itertools import combinations

# Load dataset
df = pd.read_csv("teknologi.csv")

# Definisi kriteria dan range
criteria = {
    'Scope': {'name': 'scope', 'weight': 0.140, 'range': [0, 0.16]},
    'Prospects': {'name': 'prospects', 'weight': 0.232, 'range': [0, 0.16]},
    'Potential': {'name': 'potential', 'weight': 0.175, 'range': [0, 0.16]},
    'Economy': {'name': 'economy', 'weight': 0.208, 'range': [0, 0.16]},
    'Efficiency': {'name': 'efficiency', 'weight': 0.245, 'range': [0, 0.16]}
}

# Membership Function dengan overlap
def create_manual_mf(var):
    var['low'] = fuzz.trimf(var.universe, [0.00, 0.04, 0.08])
    var['high'] = fuzz.trimf(var.universe, [0.08, 0.12, 0.16])

# Inisialisasi variabel fuzzy
input_vars = {}
output_var = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'criticality')
output_var['unsatisfactory'] = fuzz.trapmf(output_var.universe, [0, 0, 0.477, 0.477])
output_var['satisfactory'] = fuzz.trapmf(output_var.universe, [0.477, 0.477, 1, 1])

for col, info in criteria.items():
    var = ctrl.Antecedent(np.arange(info['range'][0], info['range'][1]+0.01, 0.01), info['name'])
    create_manual_mf(var)
    input_vars[col] = var

# Aturan fuzzy
rules = []

# Aturan utama dari jurnal
rules.append(ctrl.Rule(input_vars['Prospects']['high'] & input_vars['Efficiency']['high'], output_var['satisfactory']))
rules.append(ctrl.Rule(input_vars['Prospects']['low'] & input_vars['Efficiency']['low'], output_var['unsatisfactory']))

# Tambahan kombinasi 2 kriteria "high"
for combo in combinations(criteria.keys(), 2):
    rule_expr = input_vars[combo[0]]['high'] & input_vars[combo[1]]['high']
    rules.append(ctrl.Rule(rule_expr, output_var['satisfactory']))

# Kombinasi 3, 4, dan 5 input "high"
for n in [3, 4, 5]:
    for combo in combinations(criteria.keys(), n):
        rule_expr = input_vars[combo[0]]['high']
        for c in combo[1:]:
            rule_expr = rule_expr & input_vars[c]['high']
        rules.append(ctrl.Rule(rule_expr, output_var['satisfactory']))

# Aturan default agar tidak kosong
default_antecedent = input_vars['Scope']['low'] | input_vars['Scope']['high']
default_rule = ctrl.Rule(antecedent=default_antecedent, consequent=output_var['unsatisfactory'])
rules.append(default_rule)

# Sistem fuzzy
system = ctrl.ControlSystem(rules)

# Simulasi
results = []
for _, row in df.iterrows():
    sim = ctrl.ControlSystemSimulation(system)
    valid = True
    for col, info in criteria.items():
        val = row[col]
        if pd.isna(val) or not (0.0 <= val <= 0.16):
            print(f"⚠️ Invalid input for {row['Teknologi']} in column '{col}': {val}")
            valid = False
            break
        sim.input[info['name']] = val

    if not valid:
        results.append((row['Teknologi'], np.nan, "Error: invalid input"))
        continue

    try:
        sim.compute()
        crit = sim.output['criticality']
        status = "Critical" if crit >= 0.477 else "Non-Critical"
        print(f"{row['Teknologi']} ➤ Criticality: {crit:.3f} → {status}")
    except Exception as e:
        crit = np.nan
        status = f"Error: {e}"
        print(f"❌ Error computing {row['Teknologi']}: {e}")

    results.append((row['Teknologi'], crit, status))

# Simpan hasil
output_df = pd.DataFrame(results, columns=['Teknologi', 'Criticality', 'Status'])
output_df.to_csv("fuzzy_output.csv", index=False)

print("✅ Fuzzy Mamdani selesai. Hasil disimpan ke fuzzy_output.csv")
