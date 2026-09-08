# Quarterly Warehouse Inspection Report

This is a fictional, synthetic report written for a multimodal RAG demo.
Northlight Robotics is not a real company.

## Purpose

Every quarter, Northlight Robotics' inspection drones scan warehouse
shelves overnight and the Field Operations team compiles a short report
pairing the raw revenue-impact numbers with a schematic of how the scan
data actually gets from a drone to a dashboard.

## What is attached

This report bundles three quarterly revenue charts (`chart_q1_blue.png`,
`chart_q2_green.png`, `chart_q3_orange.png`) and one pipeline schematic
(`schematic_pipeline.png`) showing how images move from the drone's
camera to cloud storage. See `01-chart-notes.md` for what each chart
covers and `02-pipeline-notes.md` for the schematic.

## How to read this report

Each chart image is captioned separately at ingest time and indexed
alongside this text, so a question about "the blue chart" or "the
schematic" can retrieve the right image even if you never open the
report itself.
