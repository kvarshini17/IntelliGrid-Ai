"""
rag/knowledge_base.py
-----------------------
Purpose : The project's small, curated knowledge-base corpus for RAG
          (Phase 6/7). General educational content about building
          energy-efficiency practices, used to ground the AI Assistant's
          explanations (Phase 8+) in retrievable text rather than
          LLM-invented claims.

SOURCE POLICY (read before editing this file):
    Every document below is PROJECT-AUTHORED general educational content,
    summarizing widely-published, non-controversial building-energy
    concepts (the kind commonly described by organizations such as the
    U.S. Department of Energy, ENERGY STAR, and ASHRAE). None of these
    documents is a verified reproduction of a specific named publication,
    and none carries a verified source URL -- `source_url` is intentionally
    left as None for all of them. This is documented honestly rather than
    fabricating a citation: per the project's research-integrity
    requirement, "do not invent knowledge-base sources" and "do not present
    unsupported statements as facts" are satisfied here by being explicit
    that these are general-knowledge summaries, not primary-source
    reproductions.

Each document carries full metadata (doc_id, title, topic, source,
source_url, text) so every retrieved chunk can be traced back to exactly
which document it came from.
"""

from typing import Dict, List

DOCUMENTS: List[Dict] = [
    {
        "doc_id": "hvac_efficiency",
        "title": "HVAC Energy Efficiency",
        "topic": "HVAC energy efficiency",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Heating, ventilation, and air conditioning (HVAC) systems are typically one of the "
            "largest contributors to a commercial building's total energy consumption. Reducing "
            "unnecessary HVAC runtime is one of the most effective ways to lower energy use without "
            "requiring equipment replacement. Common efficiency practices include: scheduling HVAC "
            "operation to match building occupancy patterns rather than running continuously; setting "
            "modest, well-chosen temperature setpoints rather than extreme heating or cooling targets; "
            "maintaining equipment regularly, since dirty filters and poorly calibrated thermostats "
            "reduce efficiency; and using zoning so that only occupied areas are actively conditioned. "
            "HVAC systems that continue running in empty rooms consume energy without providing any "
            "comfort benefit to anyone, since comfort conditioning only matters where people are "
            "present. This is a widely cited rationale for linking HVAC operation to occupancy."
        ),
    },
    {
        "doc_id": "occupancy_based_hvac",
        "title": "Occupancy-Based HVAC Management",
        "topic": "occupancy-based HVAC management",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Occupancy-based HVAC management ties heating and cooling operation to whether a space is "
            "actually in use, rather than running on a fixed schedule regardless of occupancy. When a "
            "room's occupancy drops to zero, HVAC conditioning no longer serves its primary purpose -- "
            "maintaining comfort for people -- and continuing to run it is a commonly cited example of "
            "avoidable energy use. Reducing or pausing HVAC operation during confirmed unoccupied periods "
            "is one of the clearest, most explainable energy-saving actions a building management system "
            "can take, because the justification does not depend on any complex modeling: no one is "
            "present to benefit from the conditioning. This logic underlies many building-automation "
            "occupancy-sensor integrations, where HVAC setpoints are relaxed automatically once a space "
            "is detected as empty, and restored before occupants are expected to return."
        ),
    },
    {
        "doc_id": "lighting_management",
        "title": "Lighting Energy Management",
        "topic": "lighting energy management",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Lighting is a highly controllable component of building energy use because, unlike HVAC, "
            "it can be switched on or off nearly instantly with no delay in restoring full service. "
            "Leaving lighting on in an unoccupied room is one of the simplest and most commonly cited "
            "sources of avoidable energy waste, since the light provides no benefit to anyone when no "
            "one is present to use it. Common lighting-efficiency practices include automatically "
            "switching off lighting in unoccupied spaces (often via occupancy or motion sensors), using "
            "daylight where available to reduce reliance on artificial lighting during the day, and "
            "scheduling non-critical area lighting (such as corridors) to switch off during known "
            "unoccupied hours. Because lighting has essentially no warm-up or cool-down delay, it is one "
            "of the easiest loads to control based on real-time occupancy without affecting comfort when "
            "occupants return."
        ),
    },
    {
        "doc_id": "energy_conservation_building_management",
        "title": "Energy Conservation and Building Energy Management",
        "topic": "energy conservation and building energy management",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Building energy management is the practice of monitoring, analyzing, and adjusting a "
            "building's energy use to reduce waste while maintaining occupant comfort and operational "
            "needs. Core principles commonly cited in energy-conservation guidance include: measuring "
            "consumption before trying to reduce it, since interventions cannot be prioritized without "
            "first understanding where energy is actually being used; matching energy-consuming systems "
            "(HVAC, lighting, equipment) to actual demand rather than running them on fixed assumptions; "
            "eliminating standby or 'phantom' loads from equipment left on when not in use; and treating "
            "energy management as an ongoing process of monitoring and adjustment rather than a one-time "
            "project. Combining automated controls (such as occupancy-based HVAC and lighting management) "
            "with clear reporting lets building managers see the effect of changes they make, which "
            "supports continuous improvement over time."
        ),
    },
    {
        "doc_id": "appliance_management",
        "title": "Appliance Energy Management",
        "topic": "appliance energy management",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Appliances and plug-loads -- computers, refrigeration, task equipment, and similar "
            "devices -- collectively make up a meaningful share of a building's total energy use, and "
            "are often less visible in energy-management programs than HVAC and lighting. Many "
            "appliances draw power even when not actively performing a task (commonly called standby or "
            "phantom load); eliminating this waste by fully powering down non-essential equipment when a "
            "space is unoccupied is a widely cited efficiency practice. Scheduling flexible, non-urgent "
            "appliance use for lower-demand periods, and keeping equipment well maintained, are also "
            "commonly recommended. As with lighting, matching appliance operation to actual occupancy -- "
            "rather than leaving equipment running regardless of whether anyone is present -- is one of "
            "the simplest ways to reduce unnecessary consumption."
        ),
    },
    {
        "doc_id": "temperature_comfort",
        "title": "Temperature and Comfort Considerations",
        "topic": "temperature and comfort considerations",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Thermal comfort in buildings depends on multiple factors beyond a single temperature "
            "number, including humidity, airflow, and individual occupant preference, which is why "
            "comfort standards are typically expressed as a RANGE rather than one fixed ideal "
            "temperature. For general awareness: many commonly cited comfort ranges for indoor spaces "
            "fall roughly between the low-to-high 20s degrees Celsius, with a wider range for unoccupied "
            "or transitional periods. Energy-conscious temperature management typically involves setting "
            "setpoints toward the less energy-intensive end of an acceptable comfort range, rather than "
            "targeting the center of the range unnecessarily, and relaxing setpoints further when a space "
            "is confirmed unoccupied, since comfort is only a relevant consideration where people are "
            "present. This project does not adopt a specific numeric comfort standard as authoritative; "
            "any temperature thresholds used elsewhere in this project are explicitly documented as "
            "project assumptions, not engineering standards."
        ),
    },
    {
        "doc_id": "tariff_and_smart_building_practices",
        "title": "Electricity Tariff Awareness and Smart Building Practices",
        "topic": "electricity tariff awareness and general smart-building energy practices",
        "source": "Project-authored summary of general building-energy-efficiency guidance",
        "source_url": None,
        "text": (
            "Many electricity tariff structures vary by time of day, commonly dividing the day into "
            "off-peak, standard, and peak pricing bands, with peak-period electricity typically costing "
            "more due to higher grid demand. Awareness of these bands allows facility operators to shift "
            "flexible, non-time-critical energy use toward lower-cost periods where feasible, which can "
            "reduce operating cost without necessarily reducing total energy consumption. More broadly, "
            "smart-building energy practices combine several of the ideas described elsewhere in this "
            "knowledge base: occupancy-based control of HVAC and lighting, elimination of standby "
            "appliance loads, regular equipment maintenance, and ongoing monitoring to catch unusual "
            "consumption patterns. None of these practices alone eliminates energy waste; they are most "
            "effective when combined and applied consistently, with their impact measured over time "
            "rather than assumed."
        ),
    },
]


def get_documents() -> List[Dict]:
    """Return the full curated document corpus (defensive copy)."""
    return [dict(doc) for doc in DOCUMENTS]


def get_topics() -> List[str]:
    """Return the list of topics covered by the corpus."""
    return [doc["topic"] for doc in DOCUMENTS]
