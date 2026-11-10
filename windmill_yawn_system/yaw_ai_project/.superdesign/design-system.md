# Wind Turbine Yaw Control — Dashboard Design System

## Product and audience

An experimental dashboard for a wind-turbine yaw-control research project. Its users need to inspect controller behaviour, compare neural-network experiments, and distinguish proxy normalized power, physical power, and training metrics without ambiguity.

## Visual direction

Professional scientific analytics. Use a calm deep-navy base, white or near-white analysis surfaces, wind-energy teal as the primary accent, and amber only for cautions or secondary series. Prioritize legibility and data density over decoration. Avoid gradients, glass effects, neon, or marketing imagery.

## Tokens

- Primary ink: `#102A43`
- Supporting ink: `#486581`
- Canvas: `#F5F8FA`
- Surface: `#FFFFFF`
- Border: `#D9E2EC`
- Primary / wind teal: `#007C7A`
- Secondary blue: `#2563EB`
- Attention accent: `#7C3AED`
- Caution amber: `#D97706`
- Success green: `#15803D`
- Danger red: `#B91C1C`
- Font: Inter or system sans-serif; use tabular figures where available.
- Corner radius: 10px for panels, 6px for compact controls.
- Shadows: subtle only, `0 1px 3px rgba(16,42,67,.12)`.
- Spacing: 8px baseline; use 16px within cards and 24–32px between sections.

## Layout and component rules

- Use a wide, responsive dashboard canvas with a restrained header.
- Place run configuration and checkpoint controls in an initially collapsible configuration panel, not across the full page.
- Make section titles sentence case, not all caps; add one-line plain-English captions.
- Use consistent panel/card containers for trace simulation, training metrics, neural experiments, and controller comparison.
- Put the primary experiment summary in compact metric cards before detailed tables.
- Keep all original controls, data sources, CSV paths, plots, tabs, and error states usable.
- Clearly label normalized proxy power and physical power (kW/W); never visually imply that the normalized proxy is a percentage.
- Use accessible contrast, readable chart labels, and descriptive captions.

## Motion

No required animation. If Streamlit affordances animate, keep them subtle and do not delay scientific workflows.
