graph [
  directed 1
  node [
    id 0
    label "s11_pressure_drop"
    type "symptom"
    severity "high"
    dataset "cmapss"
    description "HPC outlet static pressure below normal range"
  ]
  node [
    id 1
    label "s9_speed_reduction"
    type "symptom"
    severity "high"
    dataset "cmapss"
    description "Core speed dropping below design point"
  ]
  node [
    id 2
    label "s4_temp_rise"
    type "symptom"
    severity "high"
    dataset "cmapss"
    description "LPT outlet temperature above normal"
  ]
  node [
    id 3
    label "s3_temp_rise"
    type "symptom"
    severity "medium"
    dataset "cmapss"
    description "HPC outlet temperature increasing"
  ]
  node [
    id 4
    label "s14_speed_drift"
    type "symptom"
    severity "medium"
    dataset "cmapss"
    description "Corrected core speed deviating from baseline"
  ]
  node [
    id 5
    label "s8_fan_speed_drop"
    type "symptom"
    severity "medium"
    dataset "cmapss"
    description "Physical fan speed below normal"
  ]
  node [
    id 6
    label "s13_fan_corrected_drop"
    type "symptom"
    severity "low"
    dataset "cmapss"
    description "Corrected fan speed drifting"
  ]
  node [
    id 7
    label "s21_coolant_drop"
    type "symptom"
    severity "medium"
    dataset "cmapss"
    description "LPT coolant bleed flow decreasing"
  ]
  node [
    id 8
    label "s15_bypass_drop"
    type "symptom"
    severity "medium"
    dataset "cmapss"
    description "Engine bypass ratio below specification"
  ]
  node [
    id 9
    label "hdf_heat_dissipation"
    type "symptom"
    severity "high"
    dataset "ai4i"
    description "Machine overheating &#8212; cannot cool fast enough"
  ]
  node [
    id 10
    label "pwf_power_failure"
    type "symptom"
    severity "high"
    dataset "ai4i"
    description "Machine receives wrong power level for speed"
  ]
  node [
    id 11
    label "osf_overstrain"
    type "symptom"
    severity "high"
    dataset "ai4i"
    description "Machine pushed beyond physical limit"
  ]
  node [
    id 12
    label "twf_tool_wear"
    type "symptom"
    severity "medium"
    dataset "ai4i"
    description "Tool used beyond safe wear limit"
  ]
  node [
    id 13
    label "rnf_random"
    type "symptom"
    severity "low"
    dataset "ai4i"
    description "Unexplained random failure"
  ]
  node [
    id 14
    label "secom_process_drift"
    type "symptom"
    severity "high"
    dataset "secom"
    description "Semiconductor manufacturing parameter out of control"
  ]
  node [
    id 15
    label "secom_contamination"
    type "symptom"
    severity "high"
    dataset "secom"
    description "Process contamination detected in sensor cluster"
  ]
  node [
    id 16
    label "bearing_wear"
    type "cause"
    severity "high"
    description "Engine bearing degrading &#8212; increases friction and heat"
  ]
  node [
    id 17
    label "blade_fouling"
    type "cause"
    severity "medium"
    description "Deposits on compressor blades reducing efficiency"
  ]
  node [
    id 18
    label "thermal_stress"
    type "cause"
    severity "high"
    description "Excessive heat causing component deformation"
  ]
  node [
    id 19
    label "cooling_degradation"
    type "cause"
    severity "high"
    description "Coolant flow reduced &#8212; thermal protection failing"
  ]
  node [
    id 20
    label "fuel_system_fault"
    type "cause"
    severity "high"
    description "Fuel delivery irregularity affecting combustion"
  ]
  node [
    id 21
    label "seal_degradation"
    type "cause"
    severity "medium"
    description "Internal seals worn &#8212; causing pressure leakage"
  ]
  node [
    id 22
    label "connector_fault"
    type "cause"
    severity "high"
    description "Connector impedance mismatch or physical damage"
  ]
  node [
    id 23
    label "overheating"
    type "cause"
    severity "high"
    description "Thermal management failure &#8212; temperature exceeding limits"
  ]
  node [
    id 24
    label "power_supply_fault"
    type "cause"
    severity "high"
    description "Voltage or current outside specification"
  ]
  node [
    id 25
    label "mechanical_overload"
    type "cause"
    severity "high"
    description "Forces exceeding component design limits"
  ]
  node [
    id 26
    label "tool_end_of_life"
    type "cause"
    severity "medium"
    description "Cutting tool has exceeded useful operating hours"
  ]
  node [
    id 27
    label "process_contamination"
    type "cause"
    severity "high"
    description "Foreign material in manufacturing process"
  ]
  node [
    id 28
    label "inspect_bearing"
    type "repair"
    priority "P1"
    estimated_time "4 hours"
    description "Remove and inspect bearing &#8212; replace if worn beyond limit"
  ]
  node [
    id 29
    label "clean_compressor_blades"
    type "repair"
    priority "P2"
    estimated_time "2 hours"
    description "Water wash or chemical clean compressor stage"
  ]
  node [
    id 30
    label "check_cooling_system"
    type "repair"
    priority "P1"
    estimated_time "1 hour"
    description "Inspect coolant flow, check for blockages, verify pump operation"
  ]
  node [
    id 31
    label "replace_seals"
    type "repair"
    priority "P2"
    estimated_time "6 hours"
    description "Replace worn seals in affected stage"
  ]
  node [
    id 32
    label "check_fuel_system"
    type "repair"
    priority "P1"
    estimated_time "2 hours"
    description "Check fuel nozzles, filters, and pump pressure"
  ]
  node [
    id 33
    label "thermal_inspection"
    type "repair"
    priority "P1"
    estimated_time "30 minutes"
    description "Infrared scan to identify hot spots and thermal anomalies"
  ]
  node [
    id 34
    label "retorque_connector"
    type "repair"
    priority "P1"
    estimated_time "15 minutes"
    description "Re-torque SMA/N-type connector to spec (0.9 N&#183;m)"
  ]
  node [
    id 35
    label "replace_connector"
    type "repair"
    priority "P2"
    estimated_time "30 minutes"
    description "Replace damaged connector &#8212; clean mating surfaces"
  ]
  node [
    id 36
    label "check_power_supply"
    type "repair"
    priority "P1"
    estimated_time "30 minutes"
    description "Verify DC supply voltages with DMM &#8212; check for ripple"
  ]
  node [
    id 37
    label "reduce_load"
    type "repair"
    priority "P1"
    estimated_time "immediate"
    description "Reduce speed or torque to within design envelope"
  ]
  node [
    id 38
    label "replace_tool"
    type "repair"
    priority "P1"
    estimated_time "20 minutes"
    description "Install new cutting tool &#8212; verify torque and alignment"
  ]
  node [
    id 39
    label "process_audit"
    type "repair"
    priority "P2"
    estimated_time "2 hours"
    description "Full audit of process parameters &#8212; check for contamination"
  ]
  node [
    id 40
    label "retest_at_ambient"
    type "repair"
    priority "P2"
    estimated_time "1 hour"
    description "Retest component at controlled ambient &#8212; isolate thermal drift"
  ]
  node [
    id 41
    label "escalate_to_engineering"
    type "repair"
    priority "P3"
    estimated_time "varies"
    description "Random failure &#8212; log incident and escalate for root cause analysis"
  ]
  edge [
    source 0
    target 16
    weight 0.85
    relation "indicates"
  ]
  edge [
    source 0
    target 21
    weight 0.75
    relation "indicates"
  ]
  edge [
    source 0
    target 22
    weight 0.7
    relation "indicates"
  ]
  edge [
    source 1
    target 16
    weight 0.8
    relation "indicates"
  ]
  edge [
    source 1
    target 17
    weight 0.65
    relation "indicates"
  ]
  edge [
    source 1
    target 20
    weight 0.6
    relation "indicates"
  ]
  edge [
    source 2
    target 18
    weight 0.85
    relation "indicates"
  ]
  edge [
    source 2
    target 19
    weight 0.8
    relation "indicates"
  ]
  edge [
    source 3
    target 18
    weight 0.75
    relation "indicates"
  ]
  edge [
    source 3
    target 17
    weight 0.6
    relation "indicates"
  ]
  edge [
    source 4
    target 16
    weight 0.7
    relation "indicates"
  ]
  edge [
    source 4
    target 21
    weight 0.55
    relation "indicates"
  ]
  edge [
    source 5
    target 16
    weight 0.75
    relation "indicates"
  ]
  edge [
    source 5
    target 17
    weight 0.65
    relation "indicates"
  ]
  edge [
    source 6
    target 17
    weight 0.65
    relation "indicates"
  ]
  edge [
    source 6
    target 16
    weight 0.55
    relation "indicates"
  ]
  edge [
    source 7
    target 19
    weight 0.85
    relation "indicates"
  ]
  edge [
    source 7
    target 18
    weight 0.7
    relation "indicates"
  ]
  edge [
    source 8
    target 21
    weight 0.8
    relation "indicates"
  ]
  edge [
    source 8
    target 17
    weight 0.6
    relation "indicates"
  ]
  edge [
    source 9
    target 23
    weight 0.95
    relation "indicates"
  ]
  edge [
    source 9
    target 19
    weight 0.8
    relation "indicates"
  ]
  edge [
    source 10
    target 24
    weight 0.9
    relation "indicates"
  ]
  edge [
    source 10
    target 20
    weight 0.7
    relation "indicates"
  ]
  edge [
    source 11
    target 25
    weight 0.9
    relation "indicates"
  ]
  edge [
    source 11
    target 16
    weight 0.65
    relation "indicates"
  ]
  edge [
    source 12
    target 26
    weight 0.95
    relation "indicates"
  ]
  edge [
    source 13
    target 27
    weight 0.5
    relation "indicates"
  ]
  edge [
    source 14
    target 27
    weight 0.8
    relation "indicates"
  ]
  edge [
    source 15
    target 27
    weight 0.9
    relation "indicates"
  ]
  edge [
    source 16
    target 28
    weight 0.95
    relation "requires"
  ]
  edge [
    source 16
    target 40
    weight 0.6
    relation "requires"
  ]
  edge [
    source 17
    target 29
    weight 0.9
    relation "requires"
  ]
  edge [
    source 17
    target 28
    weight 0.5
    relation "requires"
  ]
  edge [
    source 18
    target 33
    weight 0.9
    relation "requires"
  ]
  edge [
    source 18
    target 30
    weight 0.85
    relation "requires"
  ]
  edge [
    source 18
    target 40
    weight 0.75
    relation "requires"
  ]
  edge [
    source 19
    target 30
    weight 0.95
    relation "requires"
  ]
  edge [
    source 19
    target 33
    weight 0.8
    relation "requires"
  ]
  edge [
    source 20
    target 32
    weight 0.9
    relation "requires"
  ]
  edge [
    source 21
    target 31
    weight 0.85
    relation "requires"
  ]
  edge [
    source 21
    target 40
    weight 0.65
    relation "requires"
  ]
  edge [
    source 22
    target 34
    weight 0.9
    relation "requires"
  ]
  edge [
    source 22
    target 35
    weight 0.75
    relation "requires"
  ]
  edge [
    source 23
    target 30
    weight 0.95
    relation "requires"
  ]
  edge [
    source 23
    target 33
    weight 0.9
    relation "requires"
  ]
  edge [
    source 23
    target 37
    weight 0.8
    relation "requires"
  ]
  edge [
    source 24
    target 36
    weight 0.95
    relation "requires"
  ]
  edge [
    source 25
    target 37
    weight 0.95
    relation "requires"
  ]
  edge [
    source 25
    target 28
    weight 0.7
    relation "requires"
  ]
  edge [
    source 26
    target 38
    weight 0.95
    relation "requires"
  ]
  edge [
    source 27
    target 39
    weight 0.9
    relation "requires"
  ]
  edge [
    source 27
    target 41
    weight 0.7
    relation "requires"
  ]
]
