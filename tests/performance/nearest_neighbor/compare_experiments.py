#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Hardcoded paths as requested
DIR_RQ = "/home/haakno/vespa/logs/systemtests/AnnSiftRQPerfTest/sift_rq_euclidean/results/performance/"
DIR_FLOAT = "/home/haakno/vespa/logs/systemtests/AnnSiftPerfTest/sift_data_set/results/performance/"

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        metrics = {m.attrib['name']: m.text for m in root.findall('./metrics/metric')}
        params = {p.attrib['name']: p.text for p in root.findall('./parameters/parameter')}
        return params, metrics
    except:
        return None, None

def get_stats(directory):
    # Key: (target_hits, explore_hits) -> Values: {lat, recall}
    data = defaultdict(lambda: {"lat": None, "recall": None})
    if not os.path.exists(directory):
        return data
        
    for f in os.listdir(directory):
        if not f.endswith(".xml"): continue
        params, metrics = parse_xml(os.path.join(directory, f))
        if not params: continue

        th = params.get('target_hits', '0')
        eh = params.get('explore_hits', '0')
        type_ = params.get('type')
        
        key = (int(th), int(eh))
        if type_ == 'query' and params.get('algorithm') == 'hnsw':
            data[key]['lat'] = float(metrics.get('avgresponsetime', 0))
        elif type_ == 'recall':
            data[key]['recall'] = float(metrics.get('recall.avg', 0))
    return data

def generate_comparison():
    rq_data = get_stats(DIR_RQ)
    float_data = get_stats(DIR_FLOAT)

    # Get all unique (TH, EH) combinations found in HNSW tests
    all_keys = sorted(set(rq_data.keys()) | set(float_data.keys()))
    
    # Group keys by Target Hits
    by_th = defaultdict(list)
    for k in all_keys:
        by_th[k[0]].append(k)

    report = ["# Comparison: RQ vs. Float32\n"]

    for th in sorted(by_th.keys()):
        if th == 0: continue # Skip noise
        report.append(f"## Target Hits: {th}\n")
        report.append("| EH | RQ Latency | RQ Recall | Float32 Latency | Float32 Recall | Latency Gap |")
        report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for key in by_th[th]:
            eh = key[1]
            rq = rq_data[key]
            fl = float_data[key]
            
            # Skip if we don't have enough data to compare
            if not (rq['lat'] and fl['lat']): continue

            rq_lat = rq['lat']
            fl_lat = fl['lat']
            rq_rec = f"{rq['recall']:.1f}%" if rq['recall'] is not None else "-"
            fl_rec = f"{fl['recall']:.1f}%" if fl['recall'] is not None else "-"
            
            gap = rq_lat - fl_lat
            gap_str = f"{gap:+.2f} ms"
                
            report.append(f"| {eh} | {rq_lat:.2f} ms | {rq_rec} | {fl_lat:.2f} ms | {fl_rec} | {gap_str} |")
        report.append("\n")

    content = "\n".join(report)
    # Save one level up from 'performance/' so the Ruby XML parser doesn't find it
    output_path = os.path.join(os.path.dirname(DIR_RQ.rstrip('/')), "comparison_report.md")
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Report generated at: {output_path}")
    print("\n" + content)

if __name__ == "__main__":
    generate_comparison()
