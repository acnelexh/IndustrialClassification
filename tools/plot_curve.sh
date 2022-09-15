#!/usr/bin/env bash

exp="20220831_132921"
JSON_LOGS="./work_dirs/resnet50_wire/${exp}.log.json"
KEYS="loss"
TITLE="loss_plot"
OUT_FILE="loss_plot_${exp}.png"

#python tools/analysis_tools/analyze_logs.py plot_curve  \
#    ${JSON_LOGS}  \
#    [--keys ${KEYS}]  \
#    [--title ${TITLE}]  \
#    [--legend ${LEGEND}]  \
#    [--backend ${BACKEND}]  \
#    [--style ${STYLE}]  \
#    [--out ${OUT_FILE}]  \
#    [--window-size ${WINDOW_SIZE}]

python tools/analysis_tools/analyze_logs.py plot_curve  \
    ${JSON_LOGS}  \
    --keys ${KEYS}  \
    --title ${TITLE}  \
    --out ${OUT_FILE}


KEYS="accuracy_top-1"
TITLE="acc_plot"
OUT_FILE="acc_plot_${exp}.png"

python tools/analysis_tools/analyze_logs.py plot_curve  \
    ${JSON_LOGS}  \
    --keys ${KEYS}  \
    --title ${TITLE}  \
    --out ${OUT_FILE}